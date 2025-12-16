# Analysis Approach: Frozen Hints for Code Generation

## Overview

This analysis compares the **Qwen2.5-Coder-3B-Instruct** model's performance on APPS coding problems with and without frozen hints from the **Qwen2.5-3B-Instruct** model.

## The Two Approaches

### Approach 1: Coder Model Standalone (Baseline)
```
Problem → Qwen2.5-Coder-3B-Instruct → Code
```
- No hints provided
- Direct code generation
- Baseline performance

### Approach 2: Coder Model with Frozen Hints
```
Problem → Qwen2.5-3B-Instruct → Hint (frozen)
                ↓
Problem + Frozen Hint → Qwen2.5-Coder-3B-Instruct → Code
```
- Instruct model generates a hint about the approach
- Hint is "frozen" (used as-is, not modified)
- Hint is injected into the Coder model's prompt
- Coder model generates code with this guidance

## Why This Approach?

### Research Question
**Does providing frozen hints from a general instruction-following model improve a specialized code generation model's performance?**

### Motivation for Fine-Tuning
If frozen hints help, you can:
1. Generate hints for your training dataset using the Instruct model
2. Fine-tune the Coder model on (problem + hint → code) pairs
3. Create a pipeline where the Instruct model provides hints and the Coder model uses them

## What the Analysis Outputs

For each APPS problem, you get:

1. **Problem Statement**: The coding challenge from APPS
2. **Generated Hint**: What the Instruct model suggests as an approach
3. **Coder Output (No Hint)**: Code from Coder model without guidance
4. **Coder Output (With Hint)**: Code from Coder model with the frozen hint
5. **Canonical Solution**: Reference solution from APPS dataset

## Key Insights to Look For

### 1. Hint Quality
- Are hints accurate and helpful?
- Do hints capture the key algorithm/approach?
- Are hints too vague or too specific?

### 2. Hint Impact
- Does the Coder model produce better code with hints?
- Does the Coder model follow the hint's suggestions?
- When do hints help vs hurt performance?

### 3. Model Capabilities
- What kinds of problems does the Coder model solve well without hints?
- What kinds of problems benefit most from hints?
- Are there patterns by difficulty level?

### 4. Fine-Tuning Strategy
- Should you include hints in your training data?
- What format should the hints take?
- How should you structure the training examples?

## Example Output Structure

```json
{
  "problem_id": "123",
  "difficulty": "interview",
  "question": "Write a function to find the longest palindromic substring...",

  "generated_hint": "Use dynamic programming with a 2D table to track palindromes. For each substring, check if it's a palindrome by comparing characters and using previously computed results.",

  "coder_model_output_no_hint": "def longest_palindrome(s):\n    # Direct approach by Coder model...",

  "coder_model_output_with_hint": "def longest_palindrome(s):\n    # DP approach following the hint...",

  "canonical_solution": "def longest_palindrome(s):\n    # Reference solution..."
}
```

## Models Used

- **Qwen/Qwen2.5-Coder-3B-Instruct**: Specialized for code generation
  - Used for: Generating code (with and without hints)

- **Qwen/Qwen2.5-3B-Instruct**: General instruction-following model
  - Used for: Generating hints only

## Analysis Workflow

```
1. Load APPS problem
2. Generate hint using Instruct model
3. Generate code using Coder model (no hint)
4. Generate code using Coder model (with frozen hint)
5. Compare both outputs with canonical solution
6. Save all results to JSON
7. Repeat for N examples
```

## Why "Frozen" Hints?

"Frozen" means the hints are:
- Generated once by the Instruct model
- Used as-is without modification
- Not fine-tuned or optimized
- Treated as fixed input to the Coder model

This simulates a pipeline where:
1. During training: Instruct model generates hints for training data
2. During inference: Instruct model generates hints for new problems
3. Coder model is fine-tuned to work well with these hints

## Next Steps After Analysis

### If Hints Help:
1. Generate hints for larger APPS training set
2. Create training data: (problem + hint, canonical_solution)
3. Fine-tune Coder model on this data
4. Evaluate if fine-tuned model performs better

### If Hints Don't Help:
1. Analyze why hints aren't useful
2. Try different hint formats/prompts
3. Consider fine-tuning Coder model without hints
4. Or fine-tune Instruct model to generate better hints

### If Results Are Mixed:
1. Identify when hints help (problem characteristics)
2. Create conditional hint system
3. Fine-tune model to handle both scenarios
4. Develop hint quality filtering

## Metrics to Track

When reviewing results, consider:

1. **Correctness**: Does the code solve the problem?
2. **Code Quality**: Is the code well-structured and readable?
3. **Hint Alignment**: Does code with hint follow the suggested approach?
4. **Improvement**: Is code with hint better than without?
5. **Consistency**: Does the Coder model consistently use hints?

## Implementation Details

- Both models loaded in memory simultaneously
- Generation uses sampling (temperature=0.7, top_p=0.95)
- Hints limited to 150 tokens to stay concise
- Code generation limited to 512 tokens
- Results saved in JSON for detailed analysis
