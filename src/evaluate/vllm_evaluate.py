from vllm import LLM, SamplingParams
from datasets import load_dataset
from typing import Dict, Optional, List
from multiprocessing import Pool, cpu_count
from functools import partial
import torch
import gc
import os
import json
import csv
import re
import sys
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

os.environ["TOKENIZERS_PARALLELISM"] = "false"
console = Console()


def parse_code_output(text: str) -> str:
    code_pattern = r'<code>(.*?)</code>'
    matches = re.findall(code_pattern, text, re.DOTALL)

    if matches:
        code = matches[0].strip()
        return code

    code_block_pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)

    if matches:
        code = matches[0].strip()
        return code

    return text.strip()


def initialize_model(model_name: str, config: Optional[Dict] = None):
    if config is None:
        config = {}

    console.print(f"Initializing model: {model_name}")
    model = LLM(model_name, **config)
    return model


def process_data_item_humaneval(item: Dict, system_prompt: str) -> Dict:
    message_item = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": item["prompt"]
        }
    ]

    return {
        "messages": message_item,
        "prompt": item["prompt"],
        "canonical_solution": item["canonical_solution"],
        "test": item["test"],
        "task_id": item["task_id"],
        "entry_point": item["entry_point"]
    }


def process_data_item_mbpp(item: Dict, system_prompt: str) -> Dict:
    message_item = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": item["text"]
        }
    ]

    return {
        "messages": message_item,
        "text": item["text"],
        "code": item["code"],
        "test_list": item["test_list"],
        "task_id": item["task_id"],
        "test_setup_code": item.get("test_setup_code", "")
    }


def load_data_humaneval(dataset_path: str, num_workers: Optional[int] = None) -> Dict:
    console.print(f"Loading dataset: {dataset_path}")

    dataset = load_dataset(dataset_path, split="test")

    system_prompt = """You are an intelligent coder who can solve complex programming problems. Given a problem, you need to generate a solution for it.
You will be given a problem in terms of a function signature and a description of the problem. Solve the problem and provide a solution which is properly indented and formatted.

The solution should be self-contained and should not import any external libraries. It should be in `python` language.
I should be able to add the solution string to the problem file and run it.

IMPORTANT: You must wrap your code solution in <code></code> tags. For example:
<code>
def your_function(x):
    return x + 1
</code>

Only include the function implementation inside the <code></code> tags, nothing else.
"""

    if num_workers is None:
        num_workers = min(cpu_count(), len(dataset))

    console.print(f"Processing {len(dataset)} samples with {num_workers} workers...")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Processing dataset", total=len(dataset))

        process_func = partial(process_data_item_humaneval, system_prompt=system_prompt)

        with Pool(num_workers) as pool:
            processed_items = []
            for result in pool.imap(process_func, dataset, chunksize=max(1, len(dataset) // num_workers)):
                processed_items.append(result)
                progress.update(task, advance=1)

    data_dict = {
        "messages": [item["messages"] for item in processed_items],
        "prompts": [item["prompt"] for item in processed_items],
        "canonical_solution": [item["canonical_solution"] for item in processed_items],
        "test": [item["test"] for item in processed_items],
        "task_id": [item["task_id"] for item in processed_items],
        "entry_point": [item["entry_point"] for item in processed_items]
    }

    return data_dict


def load_data_mbpp(dataset_path: str, version: str = "sanitized", num_workers: Optional[int] = None) -> Dict:
    console.print(f"Loading dataset: {dataset_path} (version={version})")

    dataset = load_dataset(dataset_path, version, split="test")

    system_prompt = """You are an expert Python programmer. You will be given a task description, and you need to write a Python function to solve it.

Your solution should:
1. Be a complete, working Python function
2. Follow the task description exactly
3. Be properly indented and formatted
4. Use standard Python libraries when needed (you can import libraries)
5. Be efficient and clean

IMPORTANT: You must wrap your complete solution in <code></code> tags. For example:
<code>
def your_function(param1, param2):
    # your implementation
    return result
</code>

Include all necessary imports inside the <code></code> tags if needed.
"""

    if num_workers is None:
        num_workers = min(cpu_count(), len(dataset))

    console.print(f"Processing {len(dataset)} samples with {num_workers} workers...")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Processing dataset", total=len(dataset))

        process_func = partial(process_data_item_mbpp, system_prompt=system_prompt)

        with Pool(num_workers) as pool:
            processed_items = []
            for result in pool.imap(process_func, dataset, chunksize=max(1, len(dataset) // num_workers)):
                processed_items.append(result)
                progress.update(task, advance=1)

    data_dict = {
        "messages": [item["messages"] for item in processed_items],
        "texts": [item["text"] for item in processed_items],
        "code": [item["code"] for item in processed_items],
        "test_list": [item["test_list"] for item in processed_items],
        "task_id": [item["task_id"] for item in processed_items],
        "test_setup_code": [item["test_setup_code"] for item in processed_items]
    }

    return data_dict


def batch_inference_humaneval(
    model: LLM,
    data_dict: Dict,
    batch_size: int,
    sampling_params: SamplingParams
) -> List[Dict]:
    total_samples = len(data_dict["messages"])
    all_results = []

    console.print(f"Running inference on {total_samples} samples (batch_size={batch_size})")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Running inference", total=total_samples)

        for i in range(0, total_samples, batch_size):
            batch_messages = data_dict["messages"][i:i + batch_size]
            batch_prompts = data_dict["prompts"][i:i + batch_size]
            batch_task_ids = data_dict["task_id"][i:i + batch_size]
            batch_canonical = data_dict["canonical_solution"][i:i + batch_size]
            batch_tests = data_dict["test"][i:i + batch_size]
            batch_entry_points = data_dict["entry_point"][i:i + batch_size]

            current_batch_size = len(batch_messages)

            try:
                outputs = model.chat(
                    messages=batch_messages,
                    sampling_params=sampling_params,
                )

                for output, task_id, prompt, canonical, test, entry_point in zip(
                    outputs,
                    batch_task_ids,
                    batch_prompts,
                    batch_canonical,
                    batch_tests,
                    batch_entry_points
                ):
                    raw_output = output.outputs[0].text
                    parsed_code = parse_code_output(raw_output)

                    all_results.append({
                        "task_id": task_id,
                        "prompt": prompt,
                        "completion": parsed_code,
                        "raw_completion": raw_output,
                        "canonical_solution": canonical,
                        "test": test,
                        "entry_point": entry_point
                    })

                progress.update(task, advance=current_batch_size)

                del outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                console.print(f"[red]Error in batch {i//batch_size + 1}: {e}[/red]")
                for task_id, prompt, canonical, test, entry_point in zip(
                    batch_task_ids,
                    batch_prompts,
                    batch_canonical,
                    batch_tests,
                    batch_entry_points
                ):
                    error_msg = f"ERROR: {str(e)}"
                    all_results.append({
                        "task_id": task_id,
                        "prompt": prompt,
                        "completion": error_msg,
                        "raw_completion": error_msg,
                        "canonical_solution": canonical,
                        "test": test,
                        "entry_point": entry_point
                    })
                progress.update(task, advance=current_batch_size)
                continue

    return all_results


def batch_inference_mbpp(
    model: LLM,
    data_dict: Dict,
    batch_size: int,
    sampling_params: SamplingParams
) -> List[Dict]:
    total_samples = len(data_dict["messages"])
    all_results = []

    console.print(f"Running inference on {total_samples} samples (batch_size={batch_size})")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Running inference", total=total_samples)

        for i in range(0, total_samples, batch_size):
            batch_messages = data_dict["messages"][i:i + batch_size]
            batch_texts = data_dict["texts"][i:i + batch_size]
            batch_task_ids = data_dict["task_id"][i:i + batch_size]
            batch_code = data_dict["code"][i:i + batch_size]
            batch_test_list = data_dict["test_list"][i:i + batch_size]
            batch_test_setup = data_dict["test_setup_code"][i:i + batch_size]

            current_batch_size = len(batch_messages)

            try:
                outputs = model.chat(
                    messages=batch_messages,
                    sampling_params=sampling_params,
                )

                for output, task_id, text, code, test_list, test_setup in zip(
                    outputs,
                    batch_task_ids,
                    batch_texts,
                    batch_code,
                    batch_test_list,
                    batch_test_setup
                ):
                    raw_output = output.outputs[0].text
                    parsed_code = parse_code_output(raw_output)

                    all_results.append({
                        "task_id": task_id,
                        "text": text,
                        "completion": parsed_code,
                        "raw_completion": raw_output,
                        "code": code,
                        "test_list": test_list,
                        "test_setup_code": test_setup
                    })

                progress.update(task, advance=current_batch_size)

                del outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                console.print(f"[red]Error in batch {i//batch_size + 1}: {e}[/red]")
                for task_id, text, code, test_list, test_setup in zip(
                    batch_task_ids,
                    batch_texts,
                    batch_code,
                    batch_test_list,
                    batch_test_setup
                ):
                    error_msg = f"ERROR: {str(e)}"
                    all_results.append({
                        "task_id": task_id,
                        "text": text,
                        "completion": error_msg,
                        "raw_completion": error_msg,
                        "code": code,
                        "test_list": test_list,
                        "test_setup_code": test_setup
                    })
                progress.update(task, advance=current_batch_size)
                continue

    return all_results


def save_results_humaneval(results: List[Dict], output_folder: str, model_name: str):
    os.makedirs(output_folder, exist_ok=True)

    model_basename = model_name.replace("/", "_")
    csv_file = f"{output_folder}/{model_basename}_humaneval.csv"
    json_file = f"{output_folder}/{model_basename}_humaneval.json"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if results:
        fieldnames = ["task_id", "prompt", "completion", "raw_completion", "canonical_solution", "test", "entry_point"]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    console.print(f"\nResults saved to {json_file} and {csv_file}")


def save_results_mbpp(results: List[Dict], output_folder: str, model_name: str):
    os.makedirs(output_folder, exist_ok=True)

    model_basename = model_name.replace("/", "_")
    csv_file = f"{output_folder}/{model_basename}_mbpp.csv"
    json_file = f"{output_folder}/{model_basename}_mbpp.json"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if results:
        csv_results = []
        for r in results:
            csv_row = r.copy()
            csv_row["test_list"] = str(r["test_list"])
            csv_results.append(csv_row)

        fieldnames = ["task_id", "text", "completion", "raw_completion", "code", "test_list", "test_setup_code"]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_results)

    console.print(f"\nResults saved to {json_file} and {csv_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate code completions using VLLM"
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--data", type=str, required=True, choices=["humaneval", "mbpp"], help="Dataset to evaluate")
    parser.add_argument("--max_model_len", type=int, default=2048, help="Max model length")
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--output_folder", type=str, default="results", help="Output folder")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="sanitized",
        choices=["full", "sanitized"],
        help="Dataset version (MBPP only)"
    )

    args = parser.parse_args()

    console.print(f"\n=== VLLM Evaluation ({args.data.upper()}) ===")
    console.print(f"Model: {args.model_name}")

    try:
        model_config = {
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tensor_parallel_size
        }
        model = initialize_model(args.model_name, model_config)

        ROOT = os.path.dirname(os.path.abspath(__file__))
        CONFIG_PATH = os.path.join(ROOT, "config.json")

        params = json.load(open(CONFIG_PATH))[args.data]

        sampling_params = SamplingParams(
            temperature=params["temperature"],
            top_p=params["top_p"],
            top_k=params["top_k"],
            max_tokens=params["max_tokens"],
        )

        if args.data == "humaneval":
            dataset_name = "openai/openai_humaneval"
            data = load_data_humaneval(dataset_name, num_workers=args.num_workers)
            results = batch_inference_humaneval(model, data, args.batch_size, sampling_params)
            save_results_humaneval(results, args.output_folder, args.model_name)
        else:
            dataset_name = "Muennighoff/mbpp"
            data = load_data_mbpp(dataset_name, version=args.version, num_workers=args.num_workers)
            results = batch_inference_mbpp(model, data, args.batch_size, sampling_params)
            save_results_mbpp(results, args.output_folder, args.model_name)

        console.print(f"\n[green]✓ Evaluation completed successfully![/green]")

    except Exception as e:
        console.print(f"\n[red]✗ Error: {str(e)}[/red]")
        raise
