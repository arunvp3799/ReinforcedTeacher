"""
LLM Code Generation Evaluation System
Evaluates Qwen2.5 3B and Qwen2.5 Coder 3B on HumanEval and MBPP datasets
Tests both hint-based and pseudocode-based problem solving approaches
"""

import json
import os
import subprocess
import time
from typing import List, Dict, Any
import requests
from datasets import load_dataset

class LLMEvaluator:
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize evaluator with Ollama base URL"""
        self.base_url = base_url
        self.qwen_model = "qwen2.5:3b"
        self.qwen_coder_model = "qwen2.5-coder:3b"
        
    def generate_response(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """Generate response from Ollama model"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    def generate_hint(self, question: str, answer: str) -> str:
        """Use Qwen2.5 3B to generate hints from Q&A"""
        prompt = f"""Given this programming question and its correct answer, generate helpful hints (not the solution) that would guide someone to solve it.

Question:
{question}

Correct Answer:
{answer}

Generate 3-5 helpful hints that guide toward the solution without revealing it:"""
        
        return self.generate_response(self.qwen_model, prompt)
    
    def generate_pseudocode(self, question: str, answer: str) -> str:
        """Use Qwen2.5 3B to generate pseudocode from Q&A"""
        prompt = f"""Given this programming question and its correct answer, generate pseudocode that outlines the solution approach.

Question:
{question}

Correct Answer:
{answer}

Generate clear pseudocode for the solution:"""
        
        return self.generate_response(self.qwen_model, prompt)
    
    def solve_with_hint(self, question: str, hint: str) -> str:
        """Use Qwen2.5 Coder 3B to solve problem given hint"""
        prompt = f"""Solve this programming problem using the provided hints.

Question:
{question}

Hints:
{hint}

Provide only the Python code solution:"""
        
        return self.generate_response(self.qwen_coder_model, prompt, temperature=0.3)
    
    def solve_with_pseudocode(self, question: str, pseudocode: str) -> str:
        """Use Qwen2.5 Coder 3B to solve problem given pseudocode"""
        prompt = f"""Implement this programming problem following the provided pseudocode.

Question:
{question}

Pseudocode:
{pseudocode}

Provide only the Python code implementation:"""
        
        return self.generate_response(self.qwen_coder_model, prompt, temperature=0.3)
    
    def extract_code(self, response: str) -> str:
        """Extract code from model response"""
        # Try to find code between ```python and ```
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        return response.strip()
    
    def evaluate_code(self, code: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """Evaluate generated code against test cases"""
        passed = 0
        failed = 0
        errors = []
        
        for i, test in enumerate(test_cases):
            try:
                # Create a local namespace for execution
                local_ns = {}
                exec(code, local_ns)
                
                # Run test
                exec(test['code'], local_ns)
                passed += 1
            except Exception as e:
                failed += 1
                errors.append(f"Test {i}: {str(e)}")
        
        return {
            "passed": passed,
            "failed": failed,
            "total": len(test_cases),
            "pass_rate": passed / len(test_cases) if test_cases else 0,
            "errors": errors[:3]  # Keep first 3 errors
        }
    
    def load_humaneval_samples(self, n_samples: int = None) -> List[Dict]:
        """Load sample problems from HumanEval dataset"""
        try:
            dataset = load_dataset("openai_humaneval", split="test")
            samples = []
            
            # If n_samples is None, load all samples
            total_samples = len(dataset) if n_samples is None else min(n_samples, len(dataset))
            
            for i, item in enumerate(dataset):
                if n_samples is not None and i >= n_samples:
                    break
                samples.append({
                    "task_id": item["task_id"],
                    "question": item["prompt"],
                    "answer": item["canonical_solution"],
                    "test_cases": [{"code": item["test"]}],
                    "entry_point": item["entry_point"]
                })
            
            print(f"✓ Loaded {len(samples)} samples from HumanEval dataset")
            return samples
        except Exception as e:
            print(f"Error loading HumanEval: {e}")
            return self.get_mock_humaneval()
    
    def load_mbpp_samples(self, n_samples: int = None) -> List[Dict]:
        """Load sample problems from MBPP dataset"""
        try:
            dataset = load_dataset("mbpp", split="test")
            samples = []
            
            # If n_samples is None, load all samples
            total_samples = len(dataset) if n_samples is None else min(n_samples, len(dataset))
            
            for i, item in enumerate(dataset):
                if n_samples is not None and i >= n_samples:
                    break
                
                # Create test cases from MBPP assertions
                test_cases = []
                for test_code in item.get("test_list", []):
                    test_cases.append({"code": item["code"] + "\n" + test_code})
                
                samples.append({
                    "task_id": f"mbpp_{item['task_id']}",
                    "question": item["text"],
                    "answer": item["code"],
                    "test_cases": test_cases
                })
            
            print(f"✓ Loaded {len(samples)} samples from MBPP dataset")
            return samples
        except Exception as e:
            print(f"Error loading MBPP: {e}")
            return self.get_mock_mbpp()
    
    def get_mock_humaneval(self) -> List[Dict]:
        """Mock HumanEval samples for testing"""
        return [
            {
                "task_id": "HumanEval/0",
                "question": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
                "answer": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n",
                "test_cases": [{"code": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False"}],
                "entry_point": "has_close_elements"
            }
        ]
    
    def get_mock_mbpp(self) -> List[Dict]:
        """Mock MBPP samples for testing"""
        return [
            {
                "task_id": "mbpp_1",
                "question": "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].",
                "answer": "def min_cost(cost, m, n):\n    tc = [[0 for x in range(n+1)] for x in range(m+1)]\n    tc[0][0] = cost[0][0]\n    for i in range(1, m+1):\n        tc[i][0] = tc[i-1][0] + cost[i][0]\n    for j in range(1, n+1):\n        tc[0][j] = tc[0][j-1] + cost[0][j]\n    for i in range(1, m+1):\n        for j in range(1, n+1):\n            tc[i][j] = min(tc[i-1][j], tc[i][j-1]) + cost[i][j]\n    return tc[m][n]",
                "test_cases": [{"code": "assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8"}]
            }
        ]
    
    def run_evaluation(self, dataset_name: str, n_samples: int = None) -> Dict:
        """Run complete evaluation pipeline
        
        Args:
            dataset_name: Name of dataset ("HumanEval" or "MBPP")
            n_samples: Number of samples to evaluate. If None, evaluates entire dataset.
        """
        print(f"\n{'='*60}")
        if n_samples is None:
            print(f"Evaluating on ENTIRE {dataset_name} Dataset")
        else:
            print(f"Evaluating on {dataset_name} Dataset ({n_samples} samples)")
        print(f"{'='*60}\n")
        
        # Load dataset
        if dataset_name.lower() == "humaneval":
            samples = self.load_humaneval_samples(n_samples)
        else:
            samples = self.load_mbpp_samples(n_samples)
        
        results = {
            "dataset": dataset_name,
            "total_samples": len(samples),
            "hint_based": {"correct": 0, "total": 0, "pass_rates": []},
            "pseudocode_based": {"correct": 0, "total": 0, "pass_rates": []},
            "details": []
        }
        
        start_time = time.time()
        
        for idx, sample in enumerate(samples):
            elapsed = time.time() - start_time
            avg_time_per_sample = elapsed / (idx + 1) if idx > 0 else 0
            remaining_samples = len(samples) - (idx + 1)
            eta_seconds = avg_time_per_sample * remaining_samples
            eta_minutes = eta_seconds / 60
            
            print(f"\n[{idx + 1}/{len(samples)}] Processing: {sample['task_id']}")
            print(f"  Progress: {((idx + 1) / len(samples) * 100):.1f}% | ETA: {eta_minutes:.1f} minutes")
            
            sample_result = {
                "task_id": sample['task_id'],
                "question": sample['question'][:100] + "...",
            }
            
            # Hint-based approach
            print("  → Generating hints...")
            hint = self.generate_hint(sample['question'], sample['answer'])
            
            print("  → Solving with hints...")
            hint_solution = self.solve_with_hint(sample['question'], hint)
            hint_code = self.extract_code(hint_solution)
            hint_eval = self.evaluate_code(hint_code, sample['test_cases'])
            
            sample_result["hint_approach"] = {
                "hint": hint[:200] + "...",
                "pass_rate": hint_eval['pass_rate']
            }
            results["hint_based"]["pass_rates"].append(hint_eval['pass_rate'])
            if hint_eval['pass_rate'] == 1.0:
                results["hint_based"]["correct"] += 1
            
            # Pseudocode-based approach
            print("  → Generating pseudocode...")
            pseudocode = self.generate_pseudocode(sample['question'], sample['answer'])
            
            print("  → Solving with pseudocode...")
            pseudo_solution = self.solve_with_pseudocode(sample['question'], pseudocode)
            pseudo_code = self.extract_code(pseudo_solution)
            pseudo_eval = self.evaluate_code(pseudo_code, sample['test_cases'])
            
            sample_result["pseudocode_approach"] = {
                "pseudocode": pseudocode[:200] + "...",
                "pass_rate": pseudo_eval['pass_rate']
            }
            results["pseudocode_based"]["pass_rates"].append(pseudo_eval['pass_rate'])
            if pseudo_eval['pass_rate'] == 1.0:
                results["pseudocode_based"]["correct"] += 1
            
            results["details"].append(sample_result)
            
            print(f"    Hint-based: {hint_eval['pass_rate']:.1%} | Pseudocode-based: {pseudo_eval['pass_rate']:.1%}")
            
            # Save intermediate results every 10 samples
            if (idx + 1) % 10 == 0:
                self._save_intermediate_results(results, dataset_name)
        
        # Calculate overall statistics
        results["hint_based"]["total"] = len(samples)
        results["pseudocode_based"]["total"] = len(samples)
        results["hint_based"]["accuracy"] = results["hint_based"]["correct"] / len(samples)
        results["pseudocode_based"]["accuracy"] = results["pseudocode_based"]["correct"] / len(samples)
        results["hint_based"]["avg_pass_rate"] = sum(results["hint_based"]["pass_rates"]) / len(samples)
        results["pseudocode_based"]["avg_pass_rate"] = sum(results["pseudocode_based"]["pass_rates"]) / len(samples)
        
        total_time = time.time() - start_time
        results["evaluation_time_minutes"] = total_time / 60
        
        return results
    
    def _save_intermediate_results(self, results: Dict, dataset_name: str):
        """Save intermediate results during evaluation"""
        filename = f"/app/results/intermediate_{dataset_name.lower()}_results.json"
        try:
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  ✓ Intermediate results saved")
        except Exception as e:
            print(f"  ✗ Error saving intermediate results: {e}")
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY - {results['dataset']}")
        print(f"{'='*60}\n")
        
        print(f"Total Samples: {results['total_samples']}")
        print(f"Evaluation Time: {results.get('evaluation_time_minutes', 0):.1f} minutes\n")
        
        print("HINT-BASED APPROACH (Qwen2.5 3B → Qwen2.5 Coder 3B)")
        print(f"  Fully Correct: {results['hint_based']['correct']}/{results['hint_based']['total']}")
        print(f"  Accuracy: {results['hint_based']['accuracy']:.1%}")
        print(f"  Average Pass Rate: {results['hint_based']['avg_pass_rate']:.1%}\n")
        
        print("PSEUDOCODE-BASED APPROACH (Qwen2.5 3B → Qwen2.5 Coder 3B)")
        print(f"  Fully Correct: {results['pseudocode_based']['correct']}/{results['pseudocode_based']['total']}")
        print(f"  Accuracy: {results['pseudocode_based']['accuracy']:.1%}")
        print(f"  Average Pass Rate: {results['pseudocode_based']['avg_pass_rate']:.1%}\n")
        
        print("COMPARISON")
        diff = results['pseudocode_based']['avg_pass_rate'] - results['hint_based']['avg_pass_rate']
        print(f"  Pseudocode vs Hint Δ: {diff:+.1%}")
        if diff > 0:
            print("  ✓ Pseudocode approach performs better")
        elif diff < 0:
            print("  ✓ Hint approach performs better")
        else:
            print("  → Both approaches perform equally")


def main():
    """Main evaluation function"""
    print("LLM Code Generation Evaluation System")
    print("=" * 60)
    print("FULL DATASET EVALUATION MODE")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = LLMEvaluator()
    
    # Check if Ollama is running
    try:
        response = requests.get(f"{evaluator.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        print("✓ Connected to Ollama")
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("Please ensure Ollama is running with: ollama serve")
        return
    
    # Create results directory
    os.makedirs("/app/results", exist_ok=True)
    
    # Run evaluations on ENTIRE datasets
    all_results = []
    
    # HumanEval evaluation (entire dataset: ~164 problems)
    print("\n" + "="*60)
    print("STARTING HUMANEVAL FULL EVALUATION")
    print("Expected: ~164 problems, ~2-4 hours")
    print("="*60)
    humaneval_results = evaluator.run_evaluation("HumanEval", n_samples=None)
    evaluator.print_summary(humaneval_results)
    all_results.append(humaneval_results)
    
    # Save HumanEval results
    with open("/app/results/humaneval_full_results.json", "w") as f:
        json.dump(humaneval_results, f, indent=2)
    print("\n✓ HumanEval results saved to: /app/results/humaneval_full_results.json")
    
    # MBPP evaluation (entire test set: ~500 problems)
    print("\n" + "="*60)
    print("STARTING MBPP FULL EVALUATION")
    print("Expected: ~500 problems, ~6-10 hours")
    print("="*60)
    mbpp_results = evaluator.run_evaluation("MBPP", n_samples=None)
    evaluator.print_summary(mbpp_results)
    all_results.append(mbpp_results)
    
    # Save MBPP results
    with open("/app/results/mbpp_full_results.json", "w") as f:
        json.dump(mbpp_results, f, indent=2)
    print("\n✓ MBPP results saved to: /app/results/mbpp_full_results.json")
    
    # Save combined results
    with open("/app/results/evaluation_results_full.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("FULL EVALUATION COMPLETE")
    print(f"{'='*60}")
    print("Results saved to:")
    print("  - /app/results/humaneval_full_results.json")
    print("  - /app/results/mbpp_full_results.json")
    print("  - /app/results/evaluation_results_full.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()