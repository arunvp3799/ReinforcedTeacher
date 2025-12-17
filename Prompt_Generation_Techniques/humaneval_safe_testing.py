"""
Complete Safe HumanEval Testing System
Combines: Model Inference + Confidence Scoring + Safe Docker Execution + Analysis

Usage:
    python complete_safe_system.py --model "Qwen/Qwen2.5-Coder-3B" --output results.jsonl
    python complete_safe_system.py --analyze results.jsonl
"""

import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.data import read_problems
import docker
import tempfile
import shutil
import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
import csv


class SafeCodeExecutor:
    """Execute generated code safely in isolated Docker containers"""
    
    def __init__(self, timeout=10, memory_limit="256m"):
        """
        Initialize the safe executor
        
        Args:
            timeout: Maximum execution time in seconds
            memory_limit: Memory limit for container
        """
        self.client = docker.from_env()
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.image_name = "safe-python-executor:latest"
        self._build_docker_image()
    
    def _build_docker_image(self):
        """Build Docker image for safe execution"""
        dockerfile = '''
FROM python:3.10-slim

# Install minimal dependencies
RUN pip install --no-cache-dir numpy

# Create non-root user for security
RUN useradd -m -u 1000 coderunner
USER coderunner

# Set working directory
WORKDIR /workspace

CMD ["python", "/workspace/test.py"]
'''
        
        temp_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(temp_dir, 'Dockerfile'), 'w') as f:
                f.write(dockerfile)
            
            print("🐳 Building Docker image for safe execution...")
            self.client.images.build(
                path=temp_dir,
                tag=self.image_name,
                rm=True,
                quiet=False
            )
            print("✅ Docker image ready!")
        except docker.errors.BuildError as e:
            print(f"❌ Error building image: {e}")
            raise
        finally:
            shutil.rmtree(temp_dir)
    
    def execute_safely(self, code: str, test_code: str) -> Dict:
        """
        Execute code safely in Docker container
        
        Args:
            code: Generated code to test
            test_code: Test assertions
            
        Returns:
            Execution results dictionary
        """
        # Check for dangerous patterns
        dangerous_patterns = [
            'import os', 'import sys', 'import subprocess',
            '__import__', 'eval(', 'exec(', 'compile(',
            'open(', 'file('
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return {
                    'passed': False,
                    'error': f'Dangerous pattern blocked: {pattern}',
                    'output': None,
                    'execution_time': 0,
                    'timeout': False,
                    'is_safe': False
                }
        
        temp_dir = None
        container = None
        
        try:
            # Create test file
            temp_dir = tempfile.mkdtemp()
            test_file = os.path.join(temp_dir, 'test.py')
            
            full_code = f"""
import sys
import traceback

# Generated code
{code}

# Run tests
try:
{self._indent_code(test_code, 4)}
    print("TESTS_PASSED")
except AssertionError as e:
    print(f"ASSERTION_FAILED: {{str(e)}}")
    sys.exit(1)
except SyntaxError as e:
    print(f"SYNTAX_ERROR: {{str(e)}}")
    sys.exit(2)
except Exception as e:
    print(f"RUNTIME_ERROR: {{type(e).__name__}}: {{str(e)}}")
    traceback.print_exc()
    sys.exit(3)
"""
            
            with open(test_file, 'w') as f:
                f.write(full_code)
            
            # Run in Docker with strict limits
            start_time = time.time()
            container = self.client.containers.run(
                self.image_name,
                detach=True,
                volumes={temp_dir: {'bind': '/workspace', 'mode': 'ro'}},
                mem_limit=self.memory_limit,
                network_disabled=True,
                cpu_quota=50000,
                pids_limit=50,
                read_only=True,
                security_opt=['no-new-privileges'],
                cap_drop=['ALL']
            )
            
            # Wait with timeout
            try:
                result = container.wait(timeout=self.timeout)
                execution_time = time.time() - start_time
                logs = container.logs().decode('utf-8')
                
                # Parse results
                if "TESTS_PASSED" in logs:
                    return {
                        'passed': True,
                        'error': None,
                        'output': logs,
                        'execution_time': execution_time,
                        'timeout': False,
                        'is_safe': True
                    }
                elif "SYNTAX_ERROR" in logs:
                    return {
                        'passed': False,
                        'error': 'Syntax Error',
                        'output': logs,
                        'execution_time': execution_time,
                        'timeout': False,
                        'is_safe': True
                    }
                else:
                    return {
                        'passed': False,
                        'error': 'Runtime Error or Failed Assertion',
                        'output': logs,
                        'execution_time': execution_time,
                        'timeout': False,
                        'is_safe': True
                    }
            
            except Exception as e:
                return {
                    'passed': False,
                    'error': f'Timeout after {self.timeout}s',
                    'output': None,
                    'execution_time': self.timeout,
                    'timeout': True,
                    'is_safe': True
                }
        
        except Exception as e:
            return {
                'passed': False,
                'error': f'Execution error: {str(e)}',
                'output': None,
                'execution_time': 0,
                'timeout': False,
                'is_safe': False
            }
        
        finally:
            if container:
                try:
                    container.remove(force=True)
                except:
                    pass
            if temp_dir:
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Add indentation to code"""
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else '' 
                        for line in code.split('\n'))


class ModelInference:
    """Handle model loading and inference with confidence scoring"""
    
    def __init__(self, model_name: str):
        """
        Initialize model
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"📦 Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("✅ Model loaded successfully!")
    
    def generate_with_confidence(
        self, 
        prompt: str, 
        max_new_tokens: int = 256,
        use_cot: bool = False
    ) -> Tuple[str, float]:
        """
        Generate code with confidence score
        
        Args:
            prompt: Code generation prompt
            max_new_tokens: Maximum tokens to generate
            use_cot: Use Chain-of-Thought prompting
            
        Returns:
            Tuple of (generated_code, confidence_score)
        """
        # Apply Chain-of-Thought if requested
        if use_cot:
            enhanced_prompt = f"""Let's solve this step by step:
1. Understand the problem requirements
2. Think about edge cases
3. Write clean, correct code

{prompt}

Solution:"""
            prompt = enhanced_prompt
        
        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        # Generate with output scores for confidence
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.001,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False
        )
        
        # Calculate confidence from token probabilities
        confidences = []
        for score in outputs.scores:
            probs = F.softmax(score[0], dim=-1)
            max_prob = torch.max(probs).item()
            confidences.append(max_prob)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Decode completion
        completion = self.tokenizer.decode(
            outputs.sequences[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        
        return completion, avg_confidence


class HumanEvalTester:
    """Complete testing system combining inference and safe execution"""
    
    def __init__(self, model_name: str, timeout: int = 10):
        """
        Initialize tester
        
        Args:
            model_name: Model to test
            timeout: Execution timeout in seconds
        """
        self.model = ModelInference(model_name)
        self.executor = SafeCodeExecutor(timeout=timeout)
    
    def run_tests(
        self, 
        output_file: str = "results.jsonl",
        use_cot: bool = False,
        max_problems: int = None
    ) -> List[Dict]:
        """
        Run complete HumanEval test suite
        
        Args:
            output_file: File to save results
            use_cot: Use Chain-of-Thought prompting
            max_problems: Limit number of problems (None for all)
            
        Returns:
            List of result dictionaries
        """
        problems = read_problems()
        
        if max_problems:
            problems = dict(list(problems.items())[:max_problems])
        
        results = []
        passed_count = 0
        total_count = 0
        
        print(f"\n🧪 Testing {len(problems)} problems...")
        print(f"{'='*70}")
        
        for task_id, problem in tqdm(problems.items(), desc="Running tests"):
            # Generate code with confidence
            prompt = problem["prompt"]
            code, confidence = self.model.generate_with_confidence(
                prompt, 
                use_cot=use_cot
            )
            
            # Prepare test code
            test_code = problem["test"]
            
            # Execute safely
            exec_result = self.executor.execute_safely(code, test_code)
            
            # Aggregate results
            result = {
                'task_id': task_id,
                'completion': code,
                'confidence': confidence,
                'passed': exec_result['passed'],
                'error': exec_result.get('error'),
                'execution_time': exec_result.get('execution_time', 0),
                'timeout': exec_result.get('timeout', False),
                'is_safe': exec_result.get('is_safe', True)
            }
            
            results.append(result)
            
            if exec_result['passed']:
                passed_count += 1
            total_count += 1
            
            # Print progress every 10 problems
            if total_count % 10 == 0:
                current_accuracy = (passed_count / total_count) * 100
                tqdm.write(f"Progress: {total_count}/{len(problems)} | "
                          f"Pass rate: {current_accuracy:.1f}%")
        
        # Save results
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        # Print final summary
        self._print_summary(results, passed_count, total_count)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return results
    
    def _print_summary(self, results: List[Dict], passed: int, total: int):
        """Print test summary"""
        accuracy = (passed / total * 100) if total > 0 else 0
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        avg_exec_time = sum(r['execution_time'] for r in results) / len(results)
        
        correct_confidences = [r['confidence'] for r in results if r['passed']]
        incorrect_confidences = [r['confidence'] for r in results if not r['passed']]
        
        print("\n" + "="*70)
        print(" "*25 + "FINAL RESULTS")
        print("="*70)
        print(f"Total Problems:             {total}")
        print(f"✅ Passed:                  {passed} ({accuracy:.2f}%)")
        print(f"❌ Failed:                  {total - passed} ({100-accuracy:.2f}%)")
        print(f"\n🎯 Average Confidence:      {avg_confidence:.3f}")
        
        if correct_confidences:
            print(f"   Correct Solutions:       {sum(correct_confidences)/len(correct_confidences):.3f}")
        if incorrect_confidences:
            print(f"   Incorrect Solutions:     {sum(incorrect_confidences)/len(incorrect_confidences):.3f}")
        
        print(f"\n⏱️  Average Execution Time:  {avg_exec_time:.3f}s")
        print("="*70)


class ResultsAnalyzer:
    """Analyze and visualize testing results"""
    
    def __init__(self, results_file: str):
        """Load results from file"""
        self.results = self._load_results(results_file)
        self.results_file = results_file
    
    def _load_results(self, file_path: str) -> List[Dict]:
        """Load results from JSONL file"""
        results = []
        with open(file_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        return results
    
    def generate_report(self, output_dir: str = "analysis_results"):
        """Generate complete analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n🔍 Generating comprehensive analysis report...")
        
        # Print summary
        self._print_detailed_summary()
        
        # Generate plots
        self._plot_confidence_distribution(
            os.path.join(output_dir, "confidence_distribution.png")
        )
        self._plot_execution_time(
            os.path.join(output_dir, "execution_time.png")
        )
        self._plot_calibration(
            os.path.join(output_dir, "confidence_vs_success.png")
        )
        
        # Save detailed CSV
        csv_path = os.path.join(output_dir, "detailed_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'task_id', 'passed', 'confidence', 'execution_time', 
                'timeout', 'is_safe', 'error'
            ])
            writer.writeheader()
            for r in self.results:
                writer.writerow({
                    'task_id': r['task_id'],
                    'passed': r['passed'],
                    'confidence': f"{r['confidence']:.4f}",
                    'execution_time': f"{r['execution_time']:.4f}",
                    'timeout': r.get('timeout', False),
                    'is_safe': r.get('is_safe', True),
                    'error': r.get('error', '')
                })
        
        print(f"📄 Saved detailed CSV to: {csv_path}")
        print(f"\n✅ Analysis complete! All results saved to: {output_dir}/")
    
    def _print_detailed_summary(self):
        """Print detailed summary statistics"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        
        all_confidences = [r['confidence'] for r in self.results]
        passed_confidences = [r['confidence'] for r in self.results if r['passed']]
        failed_confidences = [r['confidence'] for r in self.results if not r['passed']]
        
        exec_times = [r['execution_time'] for r in self.results]
        timeouts = sum(1 for r in self.results if r.get('timeout', False))
        
        print("\n" + "="*70)
        print(" "*25 + "ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\n📊 Performance:")
        print(f"   Accuracy:               {(passed/total*100):.2f}%")
        print(f"   Total Problems:         {total}")
        print(f"   Passed:                 {passed}")
        print(f"   Failed:                 {total - passed}")
        
        print(f"\n🎯 Confidence Analysis:")
        print(f"   Average (All):          {np.mean(all_confidences):.3f} ± {np.std(all_confidences):.3f}")
        if passed_confidences:
            print(f"   Average (Passed):       {np.mean(passed_confidences):.3f}")
        if failed_confidences:
            print(f"   Average (Failed):       {np.mean(failed_confidences):.3f}")
        
        if passed_confidences and failed_confidences:
            conf_diff = np.mean(passed_confidences) - np.mean(failed_confidences)
            print(f"   Discrimination:         {conf_diff:.3f}")
            if conf_diff > 0.05:
                print(f"   ✅ Good calibration (higher confidence for correct answers)")
            else:
                print(f"   ⚠️  Poor calibration")
        
        print(f"\n⏱️  Execution:")
        print(f"   Average Time:           {np.mean(exec_times):.3f}s")
        print(f"   Median Time:            {np.median(exec_times):.3f}s")
        print(f"   Timeouts:               {timeouts}")
        
        print("="*70 + "\n")
    
    def _plot_confidence_distribution(self, save_path: str):
        """Plot confidence distribution"""
        passed_conf = [r['confidence'] for r in self.results if r['passed']]
        failed_conf = [r['confidence'] for r in self.results if not r['passed']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.hist(passed_conf, bins=30, alpha=0.7, label='Passed', color='green', edgecolor='black')
        ax1.hist(failed_conf, bins=30, alpha=0.7, label='Failed', color='red', edgecolor='black')
        ax1.set_xlabel('Confidence Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Confidence Distribution by Outcome', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2.boxplot([passed_conf, failed_conf], labels=['Passed', 'Failed'], patch_artist=True)
        ax2.set_ylabel('Confidence Score', fontsize=12)
        ax2.set_title('Confidence Comparison', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved confidence plot to: {save_path}")
        plt.close()
    
    def _plot_execution_time(self, save_path: str):
        """Plot execution time analysis"""
        exec_times = [r['execution_time'] for r in self.results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(exec_times, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(exec_times), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(exec_times):.3f}s')
        ax.set_xlabel('Execution Time (seconds)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Execution Time Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"⏱️  Saved execution time plot to: {save_path}")
        plt.close()
    
    def _plot_calibration(self, save_path: str):
        """Plot calibration curve"""
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        success_rates = []
        for i in range(len(bins) - 1):
            in_bin = [r for r in self.results 
                     if bins[i] <= r['confidence'] < bins[i+1]]
            if in_bin:
                success_rate = sum(1 for r in in_bin if r['passed']) / len(in_bin)
                success_rates.append(success_rate)
            else:
                success_rates.append(0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(bin_centers, success_rates, marker='o', linewidth=2, 
                markersize=8, color='blue', label='Model')
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Calibration')
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Model Calibration Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Saved calibration plot to: {save_path}")
        plt.close()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Safe HumanEval Testing System with Confidence Scoring'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='Qwen/Qwen2.5-Coder-3B',
        help='Model name from HuggingFace'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='results.jsonl',
        help='Output file for results'
    )
    parser.add_argument(
        '--cot', 
        action='store_true',
        help='Use Chain-of-Thought prompting'
    )
    parser.add_argument(
        '--max-problems', 
        type=int, 
        default=None,
        help='Maximum number of problems to test (default: all)'
    )
    parser.add_argument(
        '--timeout', 
        type=int, 
        default=10,
        help='Execution timeout in seconds'
    )
    parser.add_argument(
        '--analyze', 
        type=str,
        help='Analyze existing results file'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print(" "*15 + "SAFE HUMANEVAL TESTING SYSTEM")
    print("="*70)
    
    if args.analyze:
        # Analysis mode
        print(f"\n📊 Analysis Mode")
        print(f"Results file: {args.analyze}")
        
        analyzer = ResultsAnalyzer(args.analyze)
        analyzer.generate_report()
    else:
        # Testing mode
        print(f"\n🧪 Testing Mode")
        print(f"Model: {args.model}")
        print(f"Chain-of-Thought: {args.cot}")
        print(f"Output: {args.output}")
        print(f"Timeout: {args.timeout}s")
        print(f"Safety: Docker isolation enabled")
        print("="*70)
        
        # Run tests
        tester = HumanEvalTester(
            model_name=args.model,
            timeout=args.timeout
        )
        
        results = tester.run_tests(
            output_file=args.output,
            use_cot=args.cot,
            max_problems=args.max_problems
        )
        
        # Auto-generate analysis
        print("\n📊 Generating analysis...")
        analyzer = ResultsAnalyzer(args.output)
        analyzer.generate_report()


if __name__ == '__main__':
    main()