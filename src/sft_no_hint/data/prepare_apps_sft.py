"""
Prepare APPS Dataset for SFT (Supervised Fine-Tuning)
Target: Qwen2.5-Coder-3B

This script preprocesses the APPS dataset for standard SFT.
Input: Question
Output: Solution (Code)

Format: list of messages (ChatML style)
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "Question..."},
  {"role": "assistant", "content": "Solution..."}
]
"""

import argparse
import os
import json
from typing import Optional, List, Dict
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from datasets import Dataset

# Path to your local APPS clone
APPS_ROOT = Path("/home/ar9377/project/APPS")


def load_apps_from_extracted_files() -> Dataset:
    """
    Load APPS dataset from a local directory.
    Detects problem.json, statement.json, or question.txt for the question field.
    """
    train_dir = APPS_ROOT / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"APPS train directory not found: {train_dir}")

    print(f"Found APPS data at: {train_dir}")

    examples = []

    for problem_dir in sorted(train_dir.iterdir()):
        if not problem_dir.is_dir():
            continue

        # Detect question file
        question_file = None
        for name in ["problem.json", "statement.json", "question.txt"]:
            candidate = problem_dir / name
            if candidate.exists():
                question_file = candidate
                break

        if question_file is None:
            continue

        try:
            # Read question
            if question_file.suffix == ".txt":
                question = question_file.read_text(encoding="utf-8", errors="ignore")
                difficulty = None
            else:  # JSON files
                content = json.loads(question_file.read_text(encoding="utf-8", errors="ignore"))
                question = content.get("question") or content.get("statement") or ""
                difficulty = content.get("difficulty")

            # Read solutions
            solutions_file = problem_dir / "solutions.json"
            solutions = []
            if solutions_file.exists():
                solutions = json.loads(solutions_file.read_text(encoding="utf-8", errors="ignore"))

            # Read input/output for tests
            input_output_file = problem_dir / "input_output.json"
            input_output = {}
            if input_output_file.exists():
                input_output = json.loads(input_output_file.read_text(encoding="utf-8", errors="ignore"))

            if not question or not solutions:
                continue  # skip empty

            examples.append({
                "problem_id": problem_dir.name,
                "question": question,
                "solutions": solutions,
                "input_output": json.dumps(input_output),
                "difficulty": difficulty
            })
        except Exception as e:
            print(f"Error processing {problem_dir.name}: {e}")
            continue

    print(f"Loaded {len(examples)} examples")
    return Dataset.from_list(examples)


def extract_solution(solutions) -> Optional[str]:
    """
    Extract the first solution from the solutions list.
    """
    if isinstance(solutions, list) and len(solutions) > 0:
        return solutions[0].strip()
    return None


def format_sft_example(question: str, solution: str, system_prompt: str) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution}
        ]
    }


def prepare_apps_sft_dataset(
    output_dir: str,
    difficulty_levels: List[str] = ["introductory", "interview"],
    max_solution_length: int = 2048,
    train_split_ratio: float = 0.95,
    seed: int = 42,
    training_samples: int = -1
):
    print(f"Loading APPS dataset...")
    dataset = load_apps_from_extracted_files()

    # Filter by difficulty if available
    if difficulty_levels:
        dataset = dataset.filter(
            lambda x: (x.get("difficulty") in difficulty_levels) if x.get("difficulty") else True,
            desc=f"Filtering by difficulty: {difficulty_levels}"
        )

    print(f"Dataset size after filtering: {len(dataset)}")

    system_prompt = "You are an intelligent coding assistant. Given a programming problem, write a correct and efficient Python solution."

    processed_data = []
    skipped = 0

    for example in tqdm(dataset, desc="Processing examples"):
        question = example.get("question", "")
        solutions = example.get("solutions", [])
        solution = extract_solution(solutions)

        if not question or not solution:
            skipped += 1
            continue

        if len(solution) > max_solution_length:
            skipped += 1
            continue

        processed_data.append(format_sft_example(question, solution, system_prompt))

    print(f"Processed {len(processed_data)} examples. Skipped {skipped}.")

    df = pd.DataFrame(processed_data)
    if training_samples > 0 and len(df) > training_samples:
        df = df.sample(n=training_samples, random_state=seed)
    else:
        df = df.sample(frac=1, random_state=seed)

    train_size = int(len(df) * train_split_ratio)
    train_df = df[:train_size]
    val_df = df[train_size:]

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_json(os.path.join(output_dir, "train_sft.jsonl"), orient="records", lines=True)
    val_df.to_json(os.path.join(output_dir, "val_sft.jsonl"), orient="records", lines=True)

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./data/apps_sft")
    parser.add_argument("--samples", type=int, default=-1)
    args = parser.parse_args()

    prepare_apps_sft_dataset(
        output_dir=args.output_dir,
        training_samples=args.samples
    )


