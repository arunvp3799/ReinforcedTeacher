#!/usr/bin/env python3
"""
Generate Hints using vLLM

This script loads a teacher model and generates hints for coding problems.
The teacher sees both the question and the solution, then generates a hint
that would help a student solve the problem.

Usage:
    python scripts/generate_hints.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset humaneval
    python scripts/generate_hints.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset mbpp
"""

import argparse
import json
import os
from typing import Dict, List, Optional

from vllm import LLM, SamplingParams
from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

console = Console()


def get_hint_prompt(question: str, solution: str) -> List[Dict[str, str]]:
    """
    Create the prompt for hint generation.

    The teacher sees both the question and solution, and must generate
    a helpful hint that guides toward the solution without giving it away.
    """
    messages = [
        {
            "role": "system",
            "content": """You are an expert programming tutor. Given a coding problem and its solution, generate a helpful hint that would guide a student toward solving the problem.

Your hint should:
1. Point toward the key insight or algorithm needed
2. NOT give away the complete solution
3. Help the student understand the approach without writing the code for them
4. Be concise but informative (2-4 sentences)

Do NOT include any code in your hint. Focus on the conceptual approach."""
        },
        {
            "role": "user",
            "content": f"""Generate a helpful hint for the following problem.

<question>
{question}
</question>

<solution>
{solution}
</solution>

<hint>"""
        }
    ]
    return messages


def extract_hint(response: str) -> str:
    """Extract the hint from the model's response."""
    hint = response.strip()

    # Remove closing tag if present
    if "</hint>" in hint.lower():
        end = hint.lower().find("</hint>")
        hint = hint[:end].strip()

    return hint


def load_humaneval() -> List[Dict]:
    """Load HumanEval dataset."""
    console.print("Loading HumanEval dataset...")
    dataset = load_dataset("openai/openai_humaneval", split="test")

    items = []
    for item in dataset:
        items.append({
            "task_id": item["task_id"],
            "question": item["prompt"],
            "solution": item["canonical_solution"],
            "test": item["test"],
            "entry_point": item["entry_point"]
        })

    console.print(f"Loaded {len(items)} problems from HumanEval")
    return items


def load_mbpp(data_path: Optional[str] = None, version: str = "sanitized") -> List[Dict]:
    """Load MBPP dataset."""
    console.print(f"Loading MBPP dataset (version={version})...")

    if data_path and os.path.exists(data_path):
        # Load from local path
        if version == "sanitized":
            local_file = os.path.join(data_path, "sanitized-mbpp.json")
        else:
            local_file = os.path.join(data_path, "mbpp.jsonl")

        console.print(f"Loading from local file: {local_file}")

        dataset = []
        if local_file.endswith('.jsonl'):
            with open(local_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        dataset.append(json.loads(line))
        else:
            with open(local_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
    else:
        # Load from HuggingFace
        console.print("Loading from HuggingFace...")
        hf_dataset = load_dataset("google-research-datasets/mbpp", version, split="test")
        dataset = list(hf_dataset)

    items = []
    for item in dataset:
        # Handle different field names
        question = item.get("text", item.get("prompt", ""))
        solution = item.get("code", "")

        items.append({
            "task_id": item.get("task_id", len(items)),
            "question": question,
            "solution": solution,
            "test_list": item.get("test_list", []),
            "test_setup_code": item.get("test_setup_code", "")
        })

    console.print(f"Loaded {len(items)} problems from MBPP")
    return items


def generate_hints(
    model: LLM,
    items: List[Dict],
    sampling_params: SamplingParams,
    batch_size: int = 16
) -> List[Dict]:
    """Generate hints for all items using batch inference."""
    results = []
    total = len(items)

    console.print(f"Generating hints for {total} problems (batch_size={batch_size})")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Generating hints", total=total)

        for i in range(0, total, batch_size):
            batch_items = items[i:i + batch_size]
            batch_messages = [
                get_hint_prompt(item["question"], item["solution"])
                for item in batch_items
            ]

            try:
                outputs = model.chat(
                    messages=batch_messages,
                    sampling_params=sampling_params,
                )

                for item, output in zip(batch_items, outputs):
                    raw_hint = output.outputs[0].text
                    hint = extract_hint(raw_hint)

                    result = {
                        "task_id": item["task_id"],
                        "question": item["question"],
                        "solution": item["solution"],
                        "hint": hint,
                        "raw_hint": raw_hint
                    }

                    # Include test info if available
                    if "test" in item:
                        result["test"] = item["test"]
                        result["entry_point"] = item["entry_point"]
                    if "test_list" in item:
                        result["test_list"] = item["test_list"]
                        result["test_setup_code"] = item["test_setup_code"]

                    results.append(result)

                progress.update(task, advance=len(batch_items))

            except Exception as e:
                console.print(f"[red]Error in batch {i//batch_size + 1}: {e}[/red]")
                for item in batch_items:
                    results.append({
                        "task_id": item["task_id"],
                        "question": item["question"],
                        "solution": item["solution"],
                        "hint": f"ERROR: {str(e)}",
                        "raw_hint": f"ERROR: {str(e)}"
                    })
                progress.update(task, advance=len(batch_items))

    return results


def save_results(results: List[Dict], output_path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate hints for coding problems using vLLM"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path (e.g., Qwen/Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["humaneval", "mbpp"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--mbpp_data_path",
        type=str,
        default=None,
        help="Local path to MBPP data (optional)"
    )
    parser.add_argument(
        "--mbpp_version",
        type=str,
        default="sanitized",
        choices=["full", "sanitized"],
        help="MBPP dataset version"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/hints",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Maximum model context length"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate for hints"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0 to 1.0)"
    )

    args = parser.parse_args()

    console.print("\n=== Hint Generation with vLLM ===")
    console.print(f"Model: {args.model_name}")
    console.print(f"Dataset: {args.dataset}")

    # Initialize model
    console.print("\nInitializing model...")
    model = LLM(
        model=args.model_name,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,  # Single GPU
        trust_remote_code=True
    )

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</hint>", "\n\n\n"]
    )

    # Load dataset
    if args.dataset == "humaneval":
        items = load_humaneval()
    else:
        items = load_mbpp(args.mbpp_data_path, args.mbpp_version)

    # Generate hints
    results = generate_hints(model, items, sampling_params, args.batch_size)

    # Save results
    model_basename = args.model_name.replace("/", "_")
    output_path = os.path.join(
        args.output_dir,
        f"{model_basename}_{args.dataset}_hints.json"
    )
    save_results(results, output_path)

    # Print sample results
    console.print("\n=== Sample Results ===")
    for i, result in enumerate(results[:3]):
        console.print(f"\n[bold]Problem {i+1} (Task ID: {result['task_id']})[/bold]")
        console.print(f"[dim]Question:[/dim] {result['question'][:200]}...")
        console.print(f"[green]Hint:[/green] {result['hint']}")

    console.print(f"\n[green]✓ Generated {len(results)} hints successfully![/green]")


if __name__ == "__main__":
    main()
