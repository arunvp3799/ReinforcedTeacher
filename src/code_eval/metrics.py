"""
Pass@k metric calculation with unbiased estimator.

Based on the implementation from OpenAI's HumanEval:
https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py
"""

import itertools
from typing import Union, List, Dict, Any
import numpy as np
from collections import defaultdict


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.

    This uses the unbiased estimator: 1 - comb(n - c, k) / comb(n, k)
    where n is the total number of samples and c is the number of correct samples.

    The formula calculates the probability that at least one of k samples is correct,
    given that we observed c correct samples out of n total samples.

    Args:
        num_samples: Number of samples generated per problem (int or array)
        num_correct: Number of correct samples per problem (array)
        k: The k value for pass@k metric

    Returns:
        Array of pass@k estimates for each problem
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).

        This is mathematically equivalent to calculating the probability
        that at least one of k samples is correct.

        Edge case: If n - c < k (i.e., we have more correct samples than
        the number of incorrect samples we'd need to choose), then pass@k = 1.0
        """
        if n - c < k:
            return 1.0

        # Use product form to avoid computing large factorials
        # This computes: 1 - [(n-c)! * n! / ((n-c-k)! * (n-k)! * n!)]
        # Which simplifies to: 1 - prod(1 - k/i for i in range(n-c+1, n+1))
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def calculate_metrics(results: List[Dict[str, Any]], k_values: List[int] = [1, 10, 100]) -> Dict[str, float]:
    """
    Calculate pass@k metrics from evaluation results.

    Args:
        results: List of evaluation results, each containing:
            - task_id: Identifier for the problem
            - passed: Boolean indicating if the solution passed
        k_values: List of k values to calculate pass@k for

    Returns:
        Dictionary mapping metric names (e.g., "pass@1") to values
    """
    # Group results by task_id
    task_results = defaultdict(list)
    for result in results:
        task_id = result["task_id"]
        passed = result["passed"]
        task_results[task_id].append(passed)

    # Calculate statistics per task
    total_samples = []
    num_correct = []

    for task_id in sorted(task_results.keys()):
        results_for_task = task_results[task_id]
        total_samples.append(len(results_for_task))
        num_correct.append(sum(results_for_task))

    # Calculate pass@k for each k value
    metrics = {}
    for k in k_values:
        # Only calculate if we have enough samples
        min_samples = min(total_samples) if total_samples else 0

        if min_samples < k:
            print(f"Warning: Cannot calculate pass@{k} - minimum samples per task is {min_samples}")
            continue

        # Calculate pass@k across all tasks
        pass_at_k = estimate_pass_at_k(
            num_samples=np.array(total_samples),
            num_correct=np.array(num_correct),
            k=k,
        )

        # Average across all tasks
        metrics[f"pass@{k}"] = float(np.mean(pass_at_k))

    return metrics


def print_metrics(metrics: Dict[str, float], num_tasks: int, num_samples: int):
    """
    Pretty print evaluation metrics.

    Args:
        metrics: Dictionary of metric names to values
        num_tasks: Total number of tasks evaluated
        num_samples: Total number of samples evaluated
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total tasks evaluated: {num_tasks}")
    print(f"Total samples evaluated: {num_samples}")
    print("-" * 60)

    for metric_name in sorted(metrics.keys(), key=lambda x: int(x.split('@')[1])):
        value = metrics[metric_name]
        print(f"{metric_name:>15s}: {value:6.2%}")

    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    example_results = [
        # Task 1: 2 correct out of 10 samples
        *[{"task_id": "task_1", "passed": True}] * 2,
        *[{"task_id": "task_1", "passed": False}] * 8,
        # Task 2: 5 correct out of 10 samples
        *[{"task_id": "task_2", "passed": True}] * 5,
        *[{"task_id": "task_2", "passed": False}] * 5,
        # Task 3: 1 correct out of 10 samples
        *[{"task_id": "task_3", "passed": True}] * 1,
        *[{"task_id": "task_3", "passed": False}] * 9,
    ]

    metrics = calculate_metrics(example_results, k_values=[1, 5, 10])
    print_metrics(metrics, num_tasks=3, num_samples=30)
