"""
Evaluate LLM-generated code solutions with Docker sandboxing.

This script evaluates pre-generated solutions (JSON or CSV format)
using secure Docker-based execution and calculates pass@k metrics.

Usage:
    python evaluate_results.py \
        --input ../results/model_results.json \
        --output evaluation_results.jsonl \
        --k 1,10,100 \
        --n-workers 4
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from docker_executor import DockerExecutor
from metrics import calculate_metrics, print_metrics


def load_results_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load results from JSON file.

    Expected format:
    [
        {
            "task_id": "HumanEval/0",
            "prompt": "...",
            "completion": "...",
            "test": "...",
            "entry_point": "..."
        },
        ...
    ]

    Args:
        json_path: Path to JSON file

    Returns:
        List of problem dictionaries
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def load_results_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load results from CSV file.

    Expected columns: task_id, prompt, completion, test, entry_point

    Args:
        csv_path: Path to CSV file

    Returns:
        List of problem dictionaries
    """
    results = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            results.append(row)

    return results


def evaluate_single_solution(args: tuple) -> Dict[str, Any]:
    """
    Evaluate a single solution (for parallel processing).

    Args:
        args: Tuple of (task_id, prompt, completion, test, entry_point, timeout)

    Returns:
        Evaluation result dictionary
    """
    task_id, prompt, completion, test, entry_point, timeout = args

    executor = DockerExecutor(timeout=timeout)

    # The completion might already include the prompt, or might be just the function body
    # We need to check and handle both cases

    # If completion doesn't start with the same imports/signature as prompt, prepend prompt
    if not completion.strip().startswith(prompt.strip()[:50]):
        # Completion is just the function body, need to add prompt
        full_code = f"{prompt}{completion}"
    else:
        # Completion already includes the full function
        full_code = completion

    result = executor.check_correctness(
        task_id=task_id,
        prompt="",  # Empty since we include it in completion
        completion=full_code,
        test=test,
        entry_point=entry_point,
        timeout=timeout,
    )

    return result


def evaluate_results(
    input_path: str,
    timeout: float = 5.0,
    n_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Evaluate solutions from JSON or CSV file.

    Args:
        input_path: Path to input file (JSON or CSV)
        timeout: Timeout for each execution
        n_workers: Number of parallel workers

    Returns:
        List of evaluation results
    """
    # Detect format and load data
    input_path_obj = Path(input_path)
    suffix = input_path_obj.suffix.lower()

    print(f"Loading solutions from {input_path}...")
    if suffix == '.json':
        problems = load_results_json(input_path)
    elif suffix == '.csv':
        problems = load_results_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .json or .csv")

    print(f"Loaded {len(problems)} solutions")

    # Prepare evaluation tasks
    tasks = []
    for problem in problems:
        task_id = problem.get('task_id')
        prompt = problem.get('prompt', '')
        completion = problem.get('completion', '')
        test = problem.get('test', '')
        entry_point = problem.get('entry_point', '')

        if not task_id:
            print(f"Warning: Skipping problem without task_id")
            continue

        if not test:
            print(f"Warning: No test found for {task_id}, skipping")
            continue

        if not completion:
            print(f"Warning: No completion found for {task_id}, skipping")
            continue

        tasks.append((task_id, prompt, completion, test, entry_point, timeout))

    # Evaluate in parallel
    print(f"\nEvaluating {len(tasks)} solutions using {n_workers} workers...")
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(evaluate_single_solution, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"\nError evaluating solution: {e}")

    return results


def save_results(results: List[Dict[str, Any]], output_path: str):
    """
    Save evaluation results to a JSONL file.

    Args:
        results: List of evaluation results
        output_path: Path to output file
    """
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"\nResults saved to {output_path}")


def generate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics from results.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with summary statistics
    """
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    failed = sum(1 for r in results if not r['passed'] and 'failed' in r['result'])
    timeout = sum(1 for r in results if 'timed out' in r['result'])

    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'timed_out': timeout,
        'pass_rate': passed / total if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-generated code solutions using Docker sandboxing"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file (JSON or CSV)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output results file (default: input_name_evaluated.jsonl)",
    )
    parser.add_argument(
        "--k",
        type=str,
        default="1",
        help="Comma-separated k values for pass@k metric (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds for each execution (default: 5.0)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_evaluated.jsonl")

    # Parse k values
    k_values = [int(k.strip()) for k in args.k.split(',')]

    # Evaluate
    results = evaluate_results(
        input_path=args.input,
        timeout=args.timeout,
        n_workers=args.n_workers,
    )

    # Save results
    save_results(results, args.output)

    # Generate and print summary
    summary = generate_summary(results)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total solutions: {summary['total']}")
    print(f"Passed: {summary['passed']} ({summary['pass_rate']:.2%})")
    print(f"Failed: {summary['failed']}")
    print(f"Timed out: {summary['timed_out']}")
    print("=" * 60)

    # Calculate pass@k metrics if applicable
    # Group by task_id to see if we have multiple samples per task
    from collections import defaultdict
    task_groups = defaultdict(list)
    for r in results:
        task_groups[r['task_id']].append(r)

    # Check if we have multiple samples per task
    samples_per_task = [len(samples) for samples in task_groups.values()]
    min_samples = min(samples_per_task) if samples_per_task else 0
    max_samples = max(samples_per_task) if samples_per_task else 0

    if min_samples > 1:
        print(f"\nDetected {min_samples}-{max_samples} samples per task")
        print("Calculating pass@k metrics...")

        metrics = calculate_metrics(results, k_values=k_values)
        num_tasks = len(task_groups)
        print_metrics(metrics, num_tasks=num_tasks, num_samples=len(results))

        # Save metrics
        metrics_file = args.output.replace('.jsonl', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump({
                'metrics': metrics,
                'summary': summary,
                'num_tasks': num_tasks,
                'num_samples': len(results),
            }, f, indent=2)
        print(f"\nMetrics saved to {metrics_file}")
    else:
        print("\nOnly 1 sample per task - pass@k metrics not applicable")
        print("Use pass rate above as pass@1 metric")

        # Save summary only
        summary_file = args.output.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
