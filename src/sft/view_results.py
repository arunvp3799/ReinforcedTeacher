"""
Utility script to view and analyze the JSON results from APPS model analysis.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any


class ResultsViewer:
    def __init__(self, results_file: str):
        """Load results from JSON file."""
        self.results_file = Path(results_file)
        with open(self.results_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.metadata = self.data.get("metadata", {})
        self.results = self.data.get("results", [])

    def print_metadata(self):
        """Print metadata about the analysis."""
        print("="*80)
        print("ANALYSIS METADATA")
        print("="*80)
        for key, value in self.metadata.items():
            print(f"{key}: {value}")
        print()

    def print_example(self, example_id: int, verbose: bool = False):
        """Print details for a specific example."""
        if example_id < 1 or example_id > len(self.results):
            print(f"Error: Example ID must be between 1 and {len(self.results)}")
            return

        result = self.results[example_id - 1]

        print("="*80)
        print(f"EXAMPLE {example_id}")
        print("="*80)
        print(f"Problem ID: {result['problem_id']}")
        print(f"Difficulty: {result['difficulty']}")
        if result.get('url'):
            print(f"URL: {result['url']}")
        print()

        print("-" * 80)
        print("PROBLEM STATEMENT")
        print("-" * 80)
        print(result['question'])
        print()

        if verbose and result.get('input_output'):
            print("-" * 80)
            print("INPUT/OUTPUT EXAMPLES")
            print("-" * 80)
            print(result['input_output'])
            print()

        print("-" * 80)
        print("GENERATED HINT (from Instruct model)")
        print("-" * 80)
        print(result['generated_hint'])
        print()

        print("-" * 80)
        print("CODER MODEL OUTPUT - NO HINT (Qwen2.5-Coder-3B-Instruct)")
        print("-" * 80)
        print(result.get('coder_model_output_no_hint', result.get('coder_model_output', '')))
        print()

        print("-" * 80)
        print("CODER MODEL OUTPUT - WITH FROZEN HINT (Qwen2.5-Coder-3B-Instruct)")
        print("-" * 80)
        print(result.get('coder_model_output_with_hint', result.get('instruct_model_output', '')))
        print()

        print("-" * 80)
        print("CANONICAL SOLUTION")
        print("-" * 80)
        print(result['canonical_solution'])
        print()

        if verbose:
            print("-" * 80)
            print("CODER MODEL PROMPT (no hint)")
            print("-" * 80)
            print(result.get('coder_prompt_no_hint', result.get('coder_prompt', '')))
            print()

            print("-" * 80)
            print("CODER MODEL PROMPT (with frozen hint)")
            print("-" * 80)
            print(result.get('coder_prompt_with_hint', result.get('instruct_prompt', '')))
            print()

    def print_summary(self):
        """Print a summary of all results."""
        print("="*80)
        print("SUMMARY OF ALL EXAMPLES")
        print("="*80)
        print(f"Total examples: {len(self.results)}\n")

        for idx, result in enumerate(self.results, 1):
            print(f"{idx}. Problem ID: {result['problem_id']}")
            print(f"   Difficulty: {result['difficulty']}")
            coder_no_hint = result.get('coder_model_output_no_hint', result.get('coder_model_output', ''))
            coder_with_hint = result.get('coder_model_output_with_hint', result.get('instruct_model_output', ''))
            print(f"   Coder output (no hint): {len(coder_no_hint)} chars")
            print(f"   Coder output (with hint): {len(coder_with_hint)} chars")
            print(f"   Canonical solution length: {len(result['canonical_solution'])} chars")
            print(f"   Hint: {result['generated_hint'][:100]}...")
            print()

    def compare_outputs(self, example_id: int):
        """Side-by-side comparison of outputs."""
        if example_id < 1 or example_id > len(self.results):
            print(f"Error: Example ID must be between 1 and {len(self.results)}")
            return

        result = self.results[example_id - 1]

        print("="*80)
        print(f"COMPARISON - EXAMPLE {example_id}")
        print("="*80)
        print(f"Problem: {result['problem_id']} ({result['difficulty']})")
        print()

        outputs = [
            ("Coder Model (no hint)", result.get('coder_model_output_no_hint', result.get('coder_model_output', ''))),
            ("Coder Model (with frozen hint from Instruct)", result.get('coder_model_output_with_hint', result.get('instruct_model_output', ''))),
            ("Canonical Solution", result['canonical_solution'])
        ]

        for name, code in outputs:
            print("-" * 80)
            print(f"{name}")
            print("-" * 80)
            print(code)
            print(f"\nLength: {len(code)} characters")
            print(f"Lines: {len(code.splitlines())} lines")
            print()

    def export_to_markdown(self, output_file: str):
        """Export results to a markdown file for easy reading."""
        md_path = Path(output_file)

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# APPS Model Analysis Results\n\n")

            # Metadata
            f.write("## Metadata\n\n")
            for key, value in self.metadata.items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n---\n\n")

            # Each example
            for idx, result in enumerate(self.results, 1):
                f.write(f"## Example {idx}: {result['problem_id']}\n\n")
                f.write(f"**Difficulty**: {result['difficulty']}\n\n")

                if result.get('url'):
                    f.write(f"**URL**: {result['url']}\n\n")

                f.write("### Problem Statement\n\n")
                f.write(f"```\n{result['question']}\n```\n\n")

                f.write("### Generated Hint\n\n")
                f.write(f"{result['generated_hint']}\n\n")

                f.write("### Coder Model Output (no hint)\n\n")
                coder_no_hint = result.get('coder_model_output_no_hint', result.get('coder_model_output', ''))
                f.write(f"```python\n{coder_no_hint}\n```\n\n")

                f.write("### Coder Model Output (with frozen hint from Instruct model)\n\n")
                coder_with_hint = result.get('coder_model_output_with_hint', result.get('instruct_model_output', ''))
                f.write(f"```python\n{coder_with_hint}\n```\n\n")

                f.write("### Canonical Solution\n\n")
                f.write(f"```python\n{result['canonical_solution']}\n```\n\n")

                f.write("---\n\n")

        print(f"Results exported to: {md_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="View and analyze APPS model analysis results"
    )
    parser.add_argument(
        "results_file",
        type=str,
        nargs="?",
        default="apps_analysis_results.json",
        help="Path to results JSON file"
    )
    parser.add_argument(
        "--example",
        type=int,
        help="View specific example by ID (1-indexed)"
    )
    parser.add_argument(
        "--compare",
        type=int,
        help="Compare outputs for specific example by ID (1-indexed)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary of all examples"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output including prompts"
    )
    parser.add_argument(
        "--export-md",
        type=str,
        help="Export results to markdown file"
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        print("\nPlease run the analysis first:")
        print("  python run_analysis.py")
        return

    # Load results
    viewer = ResultsViewer(args.results_file)

    # Show metadata
    viewer.print_metadata()

    # Execute requested action
    if args.export_md:
        viewer.export_to_markdown(args.export_md)
    elif args.example:
        viewer.print_example(args.example, verbose=args.verbose)
    elif args.compare:
        viewer.compare_outputs(args.compare)
    elif args.summary:
        viewer.print_summary()
    else:
        # Default: show summary
        viewer.print_summary()
        print("\nUse --example N to view a specific example")
        print("Use --compare N to compare outputs for an example")
        print("Use --export-md FILE.md to export to markdown")


if __name__ == "__main__":
    main()
