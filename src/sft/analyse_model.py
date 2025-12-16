"""
Script to analyze Qwen2.5-Coder-3B-Instruct performance on APPS dataset with and without
frozen hints from Qwen2.5-3B-Instruct model.

This helps understand:
1. How well the Coder model performs standalone (no hints)
2. Whether frozen hints from the Instruct model improve Coder model performance
3. What kind of hints the Instruct model generates
4. How to design the training data for supervised fine-tuning

The comparison is:
- Approach 1: Coder model generates code directly (no hints)
- Approach 2: Instruct model generates hint → Coder model generates code with that hint
"""

import json
import os
from typing import List, Dict, Any
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datetime import datetime


class APPSModelAnalyzer:
    def __init__(
        self,
        coder_model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
        instruct_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = None
    ):
        """
        Initialize the analyzer with both models.

        Args:
            coder_model_name: HuggingFace model ID for the coder model
            instruct_model_name: HuggingFace model ID for the instruct model
            device: Device to use (cuda/cpu). Auto-detects if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load models and tokenizers
        print(f"\nLoading Coder model: {coder_model_name}")
        self.coder_tokenizer = AutoTokenizer.from_pretrained(coder_model_name)
        self.coder_model = AutoModelForCausalLM.from_pretrained(
            coder_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.coder_model = self.coder_model.to(self.device)

        print(f"\nLoading Instruct model: {instruct_model_name}")
        self.instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_name)
        self.instruct_model = AutoModelForCausalLM.from_pretrained(
            instruct_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.instruct_model = self.instruct_model.to(self.device)

        print("\nModels loaded successfully!")

    def load_apps_examples(self, split: str = "train", num_examples: int = 5) -> List[Dict[str, Any]]:
        """
        Load examples from APPS dataset.

        Args:
            split: Dataset split (train/test)
            num_examples: Number of examples to load

        Returns:
            List of examples with problem descriptions and solutions
        """
        print(f"\nLoading {num_examples} examples from APPS dataset ({split} split)...")
        dataset = load_dataset("codeparrot/apps", split=split)

        examples = []
        for i in range(min(num_examples, len(dataset))):
            example = dataset[i]
            examples.append({
                "problem_id": example.get("problem_id", i),
                "question": example["question"],
                "solutions": example.get("solutions", []),
                "input_output": example.get("input_output", ""),
                "difficulty": example.get("difficulty", "unknown"),
                "url": example.get("url", "")
            })

        print(f"Loaded {len(examples)} examples")
        return examples

    def create_coding_prompt(self, question: str) -> str:
        """Create a prompt for the coder model."""
        prompt = f"""You are an expert Python programmer. Write a complete Python solution for the following problem.

Problem:
{question}

Provide only the Python code without any explanations. The code should be ready to run.

Solution:
```python
"""
        return prompt

    def create_hint_prompt(self, question: str) -> str:
        """Create a prompt to generate hints for the problem."""
        prompt = f"""You are a helpful programming teacher. Analyze the following coding problem and provide a clear, concise hint about the approach to solve it. Focus on the key algorithm or data structure needed.

Problem:
{question}

Provide a brief hint (2-3 sentences) about the approach:"""
        return prompt

    def create_instruct_with_hint_prompt(self, question: str, hint: str) -> str:
        """Create a prompt for the instruct model with a hint."""
        prompt = f"""You are an expert Python programmer. Write a complete Python solution for the following problem.

Problem:
{question}

Hint: {hint}

Provide only the Python code without any explanations. The code should be ready to run.

Solution:
```python
"""
        return prompt

    def generate_code(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> str:
        """
        Generate code using the specified model.

        Args:
            model: The model to use for generation
            tokenizer: The tokenizer for the model
            prompt: The input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output
        generated_code = generated_text[len(prompt):].strip()

        return generated_code

    def extract_code_from_solution(self, solution_text: str) -> str:
        """Extract the first Python solution from the solutions JSON string."""
        try:
            solutions = json.loads(solution_text)
            if solutions and len(solutions) > 0:
                return solutions[0]
            return "No solution available"
        except:
            return solution_text if solution_text else "No solution available"

    def analyze_examples(
        self,
        examples: List[Dict[str, Any]],
        output_file: str = "apps_analysis_results.json"
    ) -> List[Dict[str, Any]]:
        """
        Analyze examples using both models and save results.

        Args:
            examples: List of APPS examples
            output_file: Path to save results JSON

        Returns:
            List of analysis results
        """
        results = []

        for idx, example in enumerate(examples):
            print(f"\n{'='*80}")
            print(f"Example {idx + 1}/{len(examples)}")
            print(f"Problem ID: {example['problem_id']}")
            print(f"Difficulty: {example['difficulty']}")
            print(f"{'='*80}")

            question = example["question"]
            canonical_solution = self.extract_code_from_solution(example["solutions"])

            print("\n--- PROBLEM ---")
            print(question[:500] + "..." if len(question) > 500 else question)

            # Generate with Coder model
            print("\n--- Generating with Qwen2.5-Coder-3B-Instruct (no hints) ---")
            coder_prompt = self.create_coding_prompt(question)
            coder_output = self.generate_code(
                self.coder_model,
                self.coder_tokenizer,
                coder_prompt
            )
            print(coder_output[:300] + "..." if len(coder_output) > 300 else coder_output)

            # Generate hint using Instruct model
            print("\n--- Generating hint with Qwen2.5-3B-Instruct ---")
            hint_prompt = self.create_hint_prompt(question)
            hint = self.generate_code(
                self.instruct_model,
                self.instruct_tokenizer,
                hint_prompt,
                max_new_tokens=150,
                temperature=0.7
            )
            print(f"Hint: {hint}")

            # Generate with Coder model using hint from Instruct model
            print("\n--- Generating with Qwen2.5-Coder-3B-Instruct (WITH frozen hint) ---")
            coder_with_hint_prompt = self.create_instruct_with_hint_prompt(question, hint)
            coder_with_hint_output = self.generate_code(
                self.coder_model,
                self.coder_tokenizer,
                coder_with_hint_prompt
            )
            print(coder_with_hint_output[:300] + "..." if len(coder_with_hint_output) > 300 else coder_with_hint_output)

            print("\n--- CANONICAL SOLUTION ---")
            print(canonical_solution[:300] + "..." if len(canonical_solution) > 300 else canonical_solution)

            # Store results
            result = {
                "example_id": idx + 1,
                "problem_id": example["problem_id"],
                "difficulty": example["difficulty"],
                "question": question,
                "url": example.get("url", ""),
                "input_output": example.get("input_output", ""),
                "canonical_solution": canonical_solution,
                "coder_model_output_no_hint": coder_output,
                "generated_hint": hint,
                "coder_model_output_with_hint": coder_with_hint_output,
                "coder_prompt_no_hint": coder_prompt,
                "coder_prompt_with_hint": coder_with_hint_prompt
            }
            results.append(result)

        # Save results to JSON
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "num_examples": len(examples),
                    "coder_model": "Qwen/Qwen2.5-Coder-3B-Instruct",
                    "instruct_model": "Qwen/Qwen2.5-3B-Instruct",
                    "device": self.device
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Results saved to: {output_path.absolute()}")
        print(f"{'='*80}")

        return results

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print a summary of the analysis."""
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total examples analyzed: {len(results)}")
        print(f"\nDifficulty distribution:")

        difficulties = {}
        for result in results:
            diff = result.get("difficulty", "unknown")
            difficulties[diff] = difficulties.get(diff, 0) + 1

        for diff, count in sorted(difficulties.items()):
            print(f"  {diff}: {count}")

        print(f"\nAverage code lengths:")
        avg_coder_no_hint = sum(len(r["coder_model_output_no_hint"]) for r in results) / len(results)
        avg_coder_with_hint = sum(len(r["coder_model_output_with_hint"]) for r in results) / len(results)
        avg_canonical = sum(len(r["canonical_solution"]) for r in results) / len(results)

        print(f"  Coder model (no hint): {avg_coder_no_hint:.0f} characters")
        print(f"  Coder model (with frozen hint): {avg_coder_with_hint:.0f} characters")
        print(f"  Canonical solutions: {avg_canonical:.0f} characters")


def main():
    """Main function to run the analysis."""
    # Configuration
    NUM_EXAMPLES = 5
    DATASET_SPLIT = "train"
    OUTPUT_FILE = "apps_analysis_results.json"

    # Initialize analyzer
    analyzer = APPSModelAnalyzer()

    # Load examples
    examples = analyzer.load_apps_examples(
        split=DATASET_SPLIT,
        num_examples=NUM_EXAMPLES
    )

    # Analyze examples
    results = analyzer.analyze_examples(
        examples=examples,
        output_file=OUTPUT_FILE
    )

    # Print summary
    analyzer.print_summary(results)

    print("\nAnalysis complete! Check the JSON file for detailed results.")


if __name__ == "__main__":
    main()
