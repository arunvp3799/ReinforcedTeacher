"""
Quick runner script for APPS model analysis with configurable parameters.
"""

import argparse
from analyse_model import APPSModelAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Qwen models on APPS dataset examples"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of examples to analyze (default: 5)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="apps_analysis_results.json",
        help="Output JSON file path (default: apps_analysis_results.json)"
    )
    parser.add_argument(
        "--coder-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
        help="HuggingFace model ID for coder model"
    )
    parser.add_argument(
        "--instruct-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID for instruct model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*80)
    print("APPS Model Analysis Configuration")
    print("="*80)
    print(f"Number of examples: {args.num_examples}")
    print(f"Dataset split: {args.split}")
    print(f"Output file: {args.output}")
    print(f"Coder model: {args.coder_model}")
    print(f"Instruct model: {args.instruct_model}")
    print(f"Device: {args.device or 'auto-detect'}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print("="*80)

    # Initialize analyzer
    analyzer = APPSModelAnalyzer(
        coder_model_name=args.coder_model,
        instruct_model_name=args.instruct_model,
        device=args.device
    )

    # Load examples
    examples = analyzer.load_apps_examples(
        split=args.split,
        num_examples=args.num_examples
    )

    # Analyze examples
    results = analyzer.analyze_examples(
        examples=examples,
        output_file=args.output
    )

    # Print summary
    analyzer.print_summary(results)

    print(f"\nAnalysis complete! Results saved to: {args.output}")
    print("\nNext steps:")
    print("1. Review the JSON file for detailed outputs")
    print("2. Compare coder vs instruct model performance")
    print("3. Analyze how hints affect the instruct model's output")
    print("4. Use insights to design your fine-tuning strategy")


if __name__ == "__main__":
    main()
