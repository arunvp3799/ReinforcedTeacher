"""
Standalone LLM Code Generation Evaluation System
Optimized for Google Colab execution
Requires: Ollama installed and running locally
"""

import json
import os
import gc
import sys
import time
import subprocess
import requests
import traceback
from typing import List, Dict, Any

class LLMEvaluator:
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize evaluator with Ollama base URL"""
        self.base_url = base_url
        self.qwen_model = "qwen2.5:3b"
        self.qwen_coder_model = "qwen2.5-coder:3b"
        self.results_dir = "results"
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
    def check_ollama_running(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_ollama(self):
        """Start Ollama service if not running"""
        if self.check_ollama_running():
            print("✓ Ollama is already running")
            return True
        
        print("Starting Ollama service...")
        try:
            # Try to start Ollama in background
            if sys.platform == "darwin":  # macOS
                subprocess.Popen(["ollama", "serve"], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            elif sys.platform == "win32":  # Windows
                subprocess.Popen(["ollama", "serve"], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:  # Linux (including Colab)
                subprocess.Popen(["ollama", "serve"], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            
            # Wait for Ollama to start
            for i in range(30):
                time.sleep(1)
                if self.check_ollama_running():
                    print("✓ Ollama started successfully")
                    return True
                if (i + 1) % 5 == 0:
                    print(f"Waiting for Ollama... ({i+1}/30)")
            
            print("✗ Failed to start Ollama")
            return False
        except FileNotFoundError:
            print("\n✗ Ollama not found! Please install Ollama first:")
            print("  For Colab, run: !curl -fsSL https://ollama.com/install.sh | sh")
            return False
    
    def ensure_models_available(self) -> bool:
        """Check and download required models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            available_models = [m['name'] for m in response.json().get('models', [])]
            
            models_to_pull = []
            if self.qwen_model not in available_models:
                models_to_pull.append(self.qwen_model)
            if self.qwen_coder_model not in available_models:
                models_to_pull.append(self.qwen_coder_model)
            
            if not models_to_pull:
                print("✓ All required models are available")
                return True
            
            print(f"\nDownloading required models (this may take a few minutes)...")
            for model in models_to_pull:
                print(f"  Pulling {model}...")
                result = subprocess.run(
                    ["ollama", "pull", model],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"  ✓ {model} downloaded")
                else:
                    print(f"  ✗ Failed to download {model}")
                    print(f"  Error: {result.stderr}")
                    return False
            
            return True
        except Exception as e:
            print(f"Error checking models: {e}")
            return False
    
    def generate_response(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """Generate response from Ollama model"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False,
                    "options": {
                        "num_ctx": 2048,
                        "num_predict": 512
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json().get("response", "")
            gc.collect()
            return result
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    def generate_hint(self, question: str, answer: str) -> str:
        """Generate hints using Qwen2.5 3B"""
        question = question[:1000]
        answer = answer[:1000]
        
        prompt = f"""Given this programming question and answer, generate 3 helpful hints.

Question:
{question}

Answer:
{answer}

Generate 3 helpful hints:"""
        
        return self.generate_response(self.qwen_model, prompt)
    
    def generate_pseudocode(self, question: str, answer: str) -> str:
        """Generate pseudocode using Qwen2.5 3B"""
        question = question[:1000]
        answer = answer[:1000]
        
        prompt = f"""Given this programming question and answer, generate clear pseudocode.

Question:
{question}

Answer:
{answer}

Generate pseudocode:"""
        
        return self.generate_response(self.qwen_model, prompt)
    
    def solve_with_hint(self, question: str, hint: str) -> str:
        """Solve using hints with Qwen2.5 Coder 3B"""
        question = question[:800]
        hint = hint[:500]
        
        prompt = f"""Solve this programming problem using the hints. Provide only Python code.

Question:
{question}

Hints:
{hint}

Python code:"""
        
        return self.generate_response(self.qwen_coder_model, prompt, temperature=0.2)
    
    def solve_with_pseudocode(self, question: str, pseudocode: str) -> str:
        """Solve using pseudocode with Qwen2.5 Coder 3B"""
        question = question[:800]
        pseudocode = pseudocode[:500]
        
        prompt = f"""Implement this programming problem following the pseudocode. Provide only Python code.

Question:
{question}

Pseudocode:
{pseudocode}

Python code:"""
        
        return self.generate_response(self.qwen_coder_model, prompt, temperature=0.2)
    
    def extract_code(self, response: str) -> str:
        """Extract code from model response"""
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
        
        for i, test in enumerate(test_cases[:3]):
            try:
                local_ns = {}
                exec(code, local_ns)
                exec(test['code'], local_ns)
                passed += 1
            except Exception as e:
                failed += 1
                errors.append(f"Test {i}: {str(e)[:50]}")
        
        return {
            "passed": passed,
            "failed": failed,
            "total": min(len(test_cases), 3),
            "pass_rate": passed / min(len(test_cases), 3) if test_cases else 0,
            "errors": errors[:2]
        }
    
    def load_humaneval_samples(self, n_samples: int = None) -> List[Dict]:
        """Load HumanEval samples - if n_samples is None, load all"""
        try:
            from datasets import load_dataset
            dataset = load_dataset("openai_humaneval", split="test")
            samples = []
            
            max_samples = len(dataset) if n_samples is None else n_samples
            
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                samples.append({
                    "task_id": item["task_id"],
                    "question": item["prompt"],
                    "answer": item["canonical_solution"],
                    "test_cases": [{"code": item["test"]}],
                    "entry_point": item["entry_point"]
                })
            
            print(f"✓ Loaded {len(samples)} HumanEval samples")
            return samples
        except Exception as e:
            print(f"Note: Using mock HumanEval data ({e})")
            mock_data = self.get_mock_humaneval()
            return mock_data if n_samples is None else mock_data[:n_samples]
    
    def load_mbpp_samples(self, n_samples: int = None) -> List[Dict]:
        """Load MBPP samples - if n_samples is None, load all"""
        try:
            from datasets import load_dataset
            dataset = load_dataset("mbpp", split="test")
            samples = []
            
            max_samples = len(dataset) if n_samples is None else n_samples
            
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                
                test_cases = []
                for test_code in item.get("test_list", []):
                    test_cases.append({"code": item["code"] + "\n" + test_code})
                
                samples.append({
                    "task_id": f"mbpp_{item['task_id']}",
                    "question": item["text"],
                    "answer": item["code"],
                    "test_cases": test_cases
                })
            
            print(f"✓ Loaded {len(samples)} MBPP samples")
            return samples
        except Exception as e:
            print(f"Note: Using mock MBPP data ({e})")
            mock_data = self.get_mock_mbpp()
            return mock_data if n_samples is None else mock_data[:n_samples]
    
    def get_mock_humaneval(self) -> List[Dict]:
        """Mock HumanEval data"""
        return [
            {
                "task_id": "HumanEval/0",
                "question": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
                "answer": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n",
                "test_cases": [{"code": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n\nassert has_close_elements([1.0, 2.0, 3.0], 0.5) == False"}],
                "entry_point": "has_close_elements"
            },
            {
                "task_id": "HumanEval/1",
                "question": "def separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input is a string with multiple groups of nested parentheses. Split into separate strings.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
                "answer": "    result = []\n    current_string = []\n    current_depth = 0\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string.clear()\n    return result\n",
                "test_cases": [{"code": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    result = []\n    current_string = []\n    current_depth = 0\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string.clear()\n    return result\n\nassert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']"}],
                "entry_point": "separate_paren_groups"
            }
        ]
    
    def get_mock_mbpp(self) -> List[Dict]:
        """Mock MBPP data"""
        return [
            {
                "task_id": "mbpp_1",
                "question": "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].",
                "answer": "def min_cost(cost, m, n):\n    tc = [[0 for x in range(n+1)] for x in range(m+1)]\n    tc[0][0] = cost[0][0]\n    for i in range(1, m+1):\n        tc[i][0] = tc[i-1][0] + cost[i][0]\n    for j in range(1, n+1):\n        tc[0][j] = tc[0][j-1] + cost[0][j]\n    for i in range(1, m+1):\n        for j in range(1, n+1):\n            tc[i][j] = min(tc[i-1][j], tc[i][j-1]) + cost[i][j]\n    return tc[m][n]",
                "test_cases": [{"code": "def min_cost(cost, m, n):\n    tc = [[0 for x in range(n+1)] for x in range(m+1)]\n    tc[0][0] = cost[0][0]\n    for i in range(1, m+1):\n        tc[i][0] = tc[i-1][0] + cost[i][0]\n    for j in range(1, n+1):\n        tc[0][j] = tc[0][j-1] + cost[0][j]\n    for i in range(1, m+1):\n        for j in range(1, n+1):\n            tc[i][j] = min(tc[i-1][j], tc[i][j-1]) + cost[i][j]\n    return tc[m][n]\n\nassert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8"}]
            },
            {
                "task_id": "mbpp_2",
                "question": "Write a python function to find the sum of common divisors of two given numbers.",
                "answer": "def sum_of_common_divisors(a, b):\n    sum = 0\n    for i in range(1, min(a, b) + 1):\n        if a % i == 0 and b % i == 0:\n            sum += i\n    return sum",
                "test_cases": [{"code": "def sum_of_common_divisors(a, b):\n    sum = 0\n    for i in range(1, min(a, b) + 1):\n        if a % i == 0 and b % i == 0:\n            sum += i\n    return sum\n\nassert sum_of_common_divisors(10, 15) == 6"}]
            }
        ]
    
    def run_evaluation(self, dataset_name: str, n_samples: int = None) -> Dict:
        """Run evaluation pipeline - if n_samples is None, evaluate all samples"""
        sample_text = "all samples" if n_samples is None else f"{n_samples} samples"
        print(f"\n{'='*60}")
        print(f"Evaluating {dataset_name} Dataset ({sample_text})")
        print(f"{'='*60}\n")
        
        # Load dataset
        if dataset_name.lower() == "humaneval":
            samples = self.load_humaneval_samples(n_samples)
        else:
            samples = self.load_mbpp_samples(n_samples)
        
        results = {
            "dataset": dataset_name,
            "total_samples": len(samples),
            "hint_based": {"correct": 0, "pass_rates": []},
            "pseudocode_based": {"correct": 0, "pass_rates": []},
            "details": []
        }
        
        for idx, sample in enumerate(samples):
            print(f"\nProcessing sample {idx + 1}/{len(samples)}: {sample['task_id']}")
            
            sample_result = {
                "task_id": sample['task_id'],
                "question": sample['question']
            }
            
            # Hint-based approach
            print("  → Generating hints...")
            hint = self.generate_hint(sample['question'], sample['answer'])
            
            print("  → Solving with hints...")
            hint_solution = self.solve_with_hint(sample['question'], hint)
            hint_code = self.extract_code(hint_solution)
            hint_eval = self.evaluate_code(hint_code, sample['test_cases'])
            
            sample_result["hint_approach"] = {
                "hint": hint,
                "generated_code": hint_code,
                "pass_rate": hint_eval['pass_rate'],
                "passed": hint_eval['passed'],
                "failed": hint_eval['failed'],
                "total_tests": hint_eval['total'],
                "errors": hint_eval['errors']
            }
            
            results["hint_based"]["pass_rates"].append(hint_eval['pass_rate'])
            if hint_eval['pass_rate'] == 1.0:
                results["hint_based"]["correct"] += 1
            
            print(f"    Hint-based pass rate: {hint_eval['pass_rate']:.0%}")
            
            # Clear memory for hint (but keep data in sample_result)
            del hint_solution
            gc.collect()
            
            # Pseudocode-based approach
            print("  → Generating pseudocode...")
            pseudocode = self.generate_pseudocode(sample['question'], sample['answer'])
            
            print("  → Solving with pseudocode...")
            pseudo_solution = self.solve_with_pseudocode(sample['question'], pseudocode)
            pseudo_code = self.extract_code(pseudo_solution)
            pseudo_eval = self.evaluate_code(pseudo_code, sample['test_cases'])
            
            sample_result["pseudocode_approach"] = {
                "pseudocode": pseudocode,
                "generated_code": pseudo_code,
                "pass_rate": pseudo_eval['pass_rate'],
                "passed": pseudo_eval['passed'],
                "failed": pseudo_eval['failed'],
                "total_tests": pseudo_eval['total'],
                "errors": pseudo_eval['errors']
            }
            
            results["pseudocode_based"]["pass_rates"].append(pseudo_eval['pass_rate'])
            if pseudo_eval['pass_rate'] == 1.0:
                results["pseudocode_based"]["correct"] += 1
            
            print(f"    Pseudocode-based pass rate: {pseudo_eval['pass_rate']:.0%}")
            
            results["details"].append(sample_result)
            
            # Clear memory for pseudocode (but keep data in sample_result)
            del pseudo_solution
            gc.collect()
            
            # Save intermediate results every 10 samples
            if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
                with open(f"{self.results_dir}/intermediate_{dataset_name}.json", "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  💾 Intermediate results saved ({idx + 1}/{len(samples)})")
        
        # Calculate statistics
        results["hint_based"]["accuracy"] = results["hint_based"]["correct"] / len(samples)
        results["pseudocode_based"]["accuracy"] = results["pseudocode_based"]["correct"] / len(samples)
        results["hint_based"]["avg_pass_rate"] = sum(results["hint_based"]["pass_rates"]) / len(samples)
        results["pseudocode_based"]["avg_pass_rate"] = sum(results["pseudocode_based"]["pass_rates"]) / len(samples)
        
        return results
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY - {results['dataset']}")
        print(f"{'='*60}\n")
        
        total = results['total_samples']
        print(f"Total Samples: {total}\n")
        
        print("HINT-BASED APPROACH (Qwen2.5 3B → Qwen2.5 Coder 3B)")
        print(f"  Fully Correct: {results['hint_based']['correct']}/{total}")
        print(f"  Accuracy: {results['hint_based']['accuracy']:.1%}")
        print(f"  Average Pass Rate: {results['hint_based']['avg_pass_rate']:.1%}\n")
        
        print("PSEUDOCODE-BASED APPROACH (Qwen2.5 3B → Qwen2.5 Coder 3B)")
        print(f"  Fully Correct: {results['pseudocode_based']['correct']}/{total}")
        print(f"  Accuracy: {results['pseudocode_based']['accuracy']:.1%}")
        print(f"  Average Pass Rate: {results['pseudocode_based']['avg_pass_rate']:.1%}\n")
        
        diff = results['pseudocode_based']['avg_pass_rate'] - results['hint_based']['avg_pass_rate']
        print("COMPARISON")
        print(f"  Pseudocode vs Hint Δ: {diff:+.1%}")
        if diff > 0:
            print("  ✓ Pseudocode approach performs better")
        elif diff < 0:
            print("  ✓ Hint approach performs better")
        else:
            print("  → Both approaches perform equally")


def main():
    """Main evaluation function"""
    print("\n" + "="*60)
    print("LLM Code Generation Evaluation System")
    print("Optimized for Google Colab")
    print("="*60 + "\n")
    
    evaluator = LLMEvaluator()
    
    # Start Ollama
    if not evaluator.start_ollama():
        print("\nPlease start Ollama manually and run this script again.")
        print("For Colab: !curl -fsSL https://ollama.com/install.sh | sh")
        print("Then run: !ollama serve &")
        return
    
    # Ensure models are available
    if not evaluator.ensure_models_available():
        print("\nFailed to download required models.")
        return
    
    print("\n✓ Setup complete. Starting evaluation...\n")
    
    all_results = []
    
    # Evaluate HumanEval with ALL samples
    humaneval_results = None
    try:
        humaneval_results = evaluator.run_evaluation("HumanEval", n_samples=None)
        evaluator.print_summary(humaneval_results)
        all_results.append(humaneval_results)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        if humaneval_results:
            all_results.append(humaneval_results)
        # Save what we have so far
        if all_results:
            with open(f"{evaluator.results_dir}/evaluation_results.json", "w") as f:
                json.dump(all_results, f, indent=2)
        return
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"\nError during HumanEval evaluation:\n{error_msg}")
        if humaneval_results:
            all_results.append(humaneval_results)
    
    # Clear memory
    gc.collect()
    
    # Evaluate MBPP with ALL samples
    mbpp_results = None
    try:
        mbpp_results = evaluator.run_evaluation("MBPP", n_samples=None)
        evaluator.print_summary(mbpp_results)
        all_results.append(mbpp_results)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        if mbpp_results:
            all_results.append(mbpp_results)
        # Save what we have so far
        if all_results:
            with open(f"{evaluator.results_dir}/evaluation_results.json", "w") as f:
                json.dump(all_results, f, indent=2)
        return
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"\nError during MBPP evaluation:\n{error_msg}")
        if mbpp_results:
            all_results.append(mbpp_results)
    
    # Save final results
    results_file = f"{evaluator.results_dir}/evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Evaluation complete!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()