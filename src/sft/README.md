# APPS Model Analysis

This directory contains scripts to analyze and compare Qwen2.5 models on the APPS dataset before fine-tuning.

## Overview

The analysis compares two approaches:
1. **Qwen2.5-Coder-3B-Instruct (no hints)**: Generates code directly without hints
2. **Qwen2.5-Coder-3B-Instruct (with frozen hints)**: Generates code with frozen hints from Qwen2.5-3B-Instruct model injected into the prompt

**How it works:**
- The Instruct model generates hints about the approach to solve each problem
- These hints are then "frozen" (used as-is) and injected into the Coder model's prompt
- This allows comparison of the Coder model's performance with and without hints

## Files

- `analyse_model.py`: Main analysis class with all functionality
- `run_analysis.py`: Command-line runner with configurable parameters
- `view_results.py`: Utility to view and compare results from JSON output
- `test_setup.py`: Quick test to verify setup before running analysis
- `requirements.txt`: Required Python packages
- `README.md`: This file

## Setup

### 1. Install Dependencies

Install required dependencies:

```bash
pip install -r requirements.txt
```

For CUDA support (recommended for faster inference):
```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
```

### 2. Verify Setup

Test your installation:

```bash
python test_setup.py
```

This will check:
- All required packages are installed
- CUDA availability (if applicable)
- Model loading works
- APPS dataset is accessible

## Usage

### Basic Usage

Run analysis with default settings (5 examples from train split):

```bash
python run_analysis.py
```

### Advanced Usage

Customize the analysis:

```bash
python run_analysis.py \
    --num-examples 10 \
    --split test \
    --output my_results.json \
    --device cuda \
    --temperature 0.8
```

### Parameters

- `--num-examples`: Number of APPS examples to analyze (default: 5)
- `--split`: Dataset split - 'train' or 'test' (default: train)
- `--output`: Output JSON file path (default: apps_analysis_results.json)
- `--coder-model`: HuggingFace model ID for coder model
- `--instruct-model`: HuggingFace model ID for instruct model
- `--device`: Device to use - 'cuda' or 'cpu' (default: auto-detect)
- `--max-tokens`: Maximum tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.7)

### Direct Python Usage

```python
from analyse_model import APPSModelAnalyzer

# Initialize
analyzer = APPSModelAnalyzer()

# Load examples
examples = analyzer.load_apps_examples(split="train", num_examples=5)

# Run analysis
results = analyzer.analyze_examples(examples, output_file="results.json")

# Print summary
analyzer.print_summary(results)
```

## Output

The script generates a JSON file with detailed results:

```json
{
  "metadata": {
    "timestamp": "2025-12-15T...",
    "num_examples": 5,
    "coder_model": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "instruct_model": "Qwen/Qwen2.5-3B-Instruct",
    "device": "cuda"
  },
  "results": [
    {
      "example_id": 1,
      "problem_id": "...",
      "difficulty": "introductory",
      "question": "...",
      "canonical_solution": "...",
      "coder_model_output_no_hint": "...",
      "generated_hint": "...",
      "coder_model_output_with_hint": "...",
      "coder_prompt_no_hint": "...",
      "coder_prompt_with_hint": "..."
    }
  ]
}
```

## What to Look For

When reviewing results:

1. **Code Quality**: Compare generated code structure and logic
2. **Hint Effectiveness**: Does the frozen hint help the Coder model perform better?
3. **Canonical Comparison**: How close are outputs to reference solutions?
4. **Hint Quality**: Are the hints from the Instruct model actually useful?
5. **Performance Difference**: Does the Coder model improve with hints?
6. **Fine-tuning Strategy**: Should you include frozen hints in your SFT training data?

## Console Output

The script prints:
- Problem statements (truncated)
- Generated hints
- Model outputs (truncated)
- Canonical solutions (truncated)
- Summary statistics

Full outputs are saved to the JSON file for detailed review.

## Performance Notes

- **Memory**: Both models loaded simultaneously (~6GB VRAM for 3B models)
- **Speed**: CUDA recommended for faster generation
- **CPU Mode**: Works but slower; reduce `--num-examples` if needed

## Viewing Results

### View Summary

```bash
python view_results.py
```

### View Specific Example

```bash
python view_results.py --example 1
```

### Compare Outputs Side-by-Side

```bash
python view_results.py --compare 1
```

### Export to Markdown

```bash
python view_results.py --export-md analysis_report.md
```

### View with Prompts (Verbose)

```bash
python view_results.py --example 1 --verbose
```

## Next Steps

After analysis:
1. Review the JSON file to understand model behaviors
2. Identify patterns in successful/unsuccessful generations
3. Design prompt templates for fine-tuning
4. Determine optimal hint injection strategy
5. Plan your supervised fine-tuning approach

## Quick Start Guide

```bash
# 1. Test your setup
python test_setup.py

# 2. Run analysis (start small)
python run_analysis.py --num-examples 3

# 3. View results
python view_results.py --summary

# 4. Deep dive into specific examples
python view_results.py --example 1 --verbose
python view_results.py --compare 1

# 5. Export for review
python view_results.py --export-md my_analysis.md
```
