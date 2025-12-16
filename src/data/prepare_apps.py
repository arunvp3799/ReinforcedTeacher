"""
Prepare APPS Dataset for RLT-Style Hint Generation Training

This script preprocesses the APPS dataset into the format required by verl.
For each coding problem, we create training examples where:
- The teacher sees: question + solution
- The teacher generates: a hint
- Reward is computed based on: P(solution | question, hint) from student model

Dataset Format for verl:
{
    "prompt": [{"role": "user", "content": "..."}],  # Messages for teacher
    "data_source": "apps",                           # For reward function routing
    "reward_model": {                                # Ground truth for reward
        "style": "custom",
        "ground_truth": {"question": "...", "solution": "..."}
    },
    "extra_info": {...}                              # Additional metadata
}
"""

import argparse
import os
import json
from typing import Optional

import datasets
import pandas as pd


def create_teacher_prompt(question: str, solution: str) -> str:
    """
    Create the prompt for the teacher model.
    The teacher sees both the question and solution, and must generate a hint.
    Uses HTML-style tags for clear structure.
    """
    prompt = f"""Given the question and answer, generate a helpful hint.

<question>
{question}
</question>

<answer>
{solution}
</answer>

Generate a hint that helps solve this problem without giving away the solution.

<hint>"""
    return prompt


def extract_solution(solutions_str: str) -> Optional[str]:
    """
    Extract the first valid Python solution from the APPS solutions field.
    APPS stores solutions as a JSON string containing a list of solutions.
    """
    try:
        solutions = json.loads(solutions_str)
        if isinstance(solutions, list) and len(solutions) > 0:
            # Return the first solution
            return solutions[0].strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def process_apps_example(example: dict, idx: int, split: str) -> Optional[dict]:
    """
    Process a single APPS example into the verl format.
    """
    question = example.get("question", "")
    solutions_str = example.get("solutions", "[]")
    difficulty = example.get("difficulty", "unknown")
    problem_id = example.get("problem_id", idx)

    # Extract a solution
    solution = extract_solution(solutions_str)
    if solution is None or len(solution.strip()) == 0:
        return None

    if len(question.strip()) == 0:
        return None

    # Create the teacher's prompt
    teacher_prompt = create_teacher_prompt(question, solution)

    # Build the data in verl format
    data = {
        "data_source": "apps_hints",
        "prompt": [
            {
                "role": "user",
                "content": teacher_prompt
            }
        ],
        "reward_model": {
            "style": "custom",
            "ground_truth": {
                "question": question,
                "solution": solution
            }
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "problem_id": problem_id,
            "difficulty": difficulty,
            "question": question,
            "solution": solution
        }
    }

    return data


def prepare_apps_dataset(
    output_dir: str,
    difficulty_levels: list[str] = ["introductory", "interview"],
    max_solution_length: int = 2000,
    max_question_length: int = 3000,
    train_split_ratio: float = 0.9,
    seed: int = 42,
    training_samples: int = 1000,
):
    """
    Download and preprocess the APPS dataset.

    Args:
        output_dir: Directory to save the processed parquet files
        difficulty_levels: Which difficulty levels to include
        max_solution_length: Maximum solution length in characters
        max_question_length: Maximum question length in characters
        train_split_ratio: Ratio of data for training vs validation
        seed: Random seed for splitting
        training_samples: Total number of samples to use (default 1000, -1 for all)
    """
    print(f"Loading APPS dataset...")

    # Load APPS dataset from HuggingFace
    # APPS has train and test splits
    dataset = datasets.load_dataset("codeparrot/apps", split="train", trust_remote_code=True)

    print(f"Total examples in APPS train: {len(dataset)}")

    # Filter by difficulty
    if difficulty_levels:
        dataset = dataset.filter(
            lambda x: x.get("difficulty", "") in difficulty_levels,
            desc=f"Filtering by difficulty: {difficulty_levels}"
        )
        print(f"After difficulty filter: {len(dataset)}")

    # Process examples
    processed_data = []
    skipped = 0

    for idx, example in enumerate(dataset):
        # Skip very long solutions or questions
        solutions_str = example.get("solutions", "[]")
        solution = extract_solution(solutions_str)
        question = example.get("question", "")

        if solution is None:
            skipped += 1
            continue

        if len(solution) > max_solution_length:
            skipped += 1
            continue

        if len(question) > max_question_length:
            skipped += 1
            continue

        processed = process_apps_example(example, idx, "train")
        if processed is not None:
            processed_data.append(processed)

    print(f"Processed {len(processed_data)} examples, skipped {skipped}")

    # Convert to DataFrame and split
    df = pd.DataFrame(processed_data)

    # Shuffle and limit to training_samples
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if training_samples > 0 and len(df) > training_samples:
        df = df[:training_samples]
        print(f"Limited to {training_samples} samples")

    train_size = int(len(df) * train_split_ratio)

    train_df = df[:train_size]
    val_df = df[train_size:]

    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

    # Save to parquet
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Saved train data to: {train_path}")
    print(f"Saved val data to: {val_path}")

    # Print sample
    print("\n" + "="*50)
    print("Sample data point:")
    print("="*50)
    sample = train_df.iloc[0].to_dict()
    print(f"data_source: {sample['data_source']}")
    print(f"prompt (first 500 chars): {str(sample['prompt'])[:500]}...")
    print(f"extra_info keys: {sample['extra_info'].keys() if isinstance(sample['extra_info'], dict) else 'N/A'}")

    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="Prepare APPS dataset for RLT hint training")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/apps_hints",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        nargs="+",
        default=["introductory", "interview"],
        choices=["introductory", "interview", "competition"],
        help="Difficulty levels to include"
    )
    parser.add_argument(
        "--max_solution_length",
        type=int,
        default=2000,
        help="Maximum solution length in characters"
    )
    parser.add_argument(
        "--max_question_length",
        type=int,
        default=3000,
        help="Maximum question length in characters"
    )
    parser.add_argument(
        "--train_split_ratio",
        type=float,
        default=0.9,
        help="Ratio of data for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--training_samples",
        type=int,
        default=1000,
        help="Total number of samples to use (-1 for all)"
    )

    args = parser.parse_args()

    prepare_apps_dataset(
        output_dir=args.output_dir,
        difficulty_levels=args.difficulty,
        max_solution_length=args.max_solution_length,
        max_question_length=args.max_question_length,
        train_split_ratio=args.train_split_ratio,
        seed=args.seed,
        training_samples=args.training_samples
    )


if __name__ == "__main__":
    main()
