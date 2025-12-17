# LLM Code Generation Evaluation System

A comprehensive evaluation system that tests Qwen2.5 3B and Qwen2.5 Coder 3B models on HumanEval and MBPP datasets using two different approaches:
1. **Hint-based**: Generate hints → Solve with hints
2. **Pseudocode-based**: Generate pseudocode → Implement from pseudocode

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Load Dataset (HumanEval/MBPP)                      │
│           ↓                                                  │
│  Step 2: Qwen2.5 3B generates hints/pseudocode              │
│           ↓                                                  │
│  Step 3: Qwen2.5 Coder 3B solves using hints/pseudocode    │
│           ↓                                                  │
│  Step 4: Execute & evaluate against test cases              │
│           ↓                                                  │
│  Step 5: Compare both approaches                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM available
- Internet connection (for downloading models and datasets)

## Quick Start

### 1. Setup Project Structure

Create the following directory structure:

```
llm-evaluator/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── evaluator.py
├── run_evaluation.sh
├── README.md
└── results/          # Created automatically
```

### 2. Build the Docker Image

```bash
docker-compose build
```

This will:
- Set up Python 3.10 environment
- Install Ollama
- Install required Python packages
- Configure the evaluation system

### 3. Run the Evaluation

```bash
docker-compose up
```

The system will:
1. Start Ollama service
2. Download Qwen2.5 3B and Qwen2.5 Coder 3B models (~4GB total)
3. Load HumanEval and MBPP datasets
4. Run evaluations on both datasets
5. Save results to `./results/evaluation_results.json`

### 4. Monitor Progress

You'll see output like:

```
============================================================
Evaluating on HumanEval Dataset (5 samples)
============================================================

Processing sample 1/5: HumanEval/0
  → Generating hints...
  → Solving with hints...
  → Generating pseudocode...
  → Solving with pseudocode...
    Hint-based pass rate: 100.0%
    Pseudocode-based pass rate: 100.0%
```

## Configuration

### Adjust Sample Size

Edit `evaluator.py` and modify the `n_samples` parameter:

```python
# In main() function
humaneval_results = evaluator.run_evaluation("HumanEval", n_samples=10)
mbpp_results = evaluator.run_evaluation("MBPP", n_samples=10)
```

### Change Models

Modify the models in `evaluator.py`:

```python
def __init__(self, base_url: str = "http://localhost:11434"):
    self.qwen_model = "qwen2.5:3b"  # Change this
    self.qwen_coder_model = "qwen2.5-coder:3b"  # Change this
```

### Adjust Temperature

Control randomness in generation:

```python
# Lower temperature (0.1-0.5) for more deterministic outputs
# Higher temperature (0.7-1.0) for more creative outputs
self.generate_response(model, prompt, temperature=0.3)
```

## Output

### Console Output

Real-time evaluation progress and summary statistics:

```
EVALUATION SUMMARY - HumanEval
============================================================

Total Samples: 5

HINT-BASED APPROACH (Qwen2.5 3B → Qwen2.5 Coder 3B)
  Fully Correct: 3/5
  Accuracy: 60.0%
  Average Pass Rate: 75.0%

PSEUDOCODE-BASED APPROACH (Qwen2.5 3B → Qwen2.5 Coder 3B)
  Fully Correct: 4/5
  Accuracy: 80.0%
  Average Pass Rate: 85.0%

COMPARISON
  Pseudocode vs Hint Δ: +10.0%
  ✓ Pseudocode approach performs better
```

### JSON Results

Detailed results saved to `./results/evaluation_results.json`:

```json
[
  {
    "dataset": "HumanEval",
    "total_samples": 5,
    "hint_based": {
      "correct": 3,
      "total": 5,
      "accuracy": 0.6,
      "avg_pass_rate": 0.75,
      "pass_rates": [1.0, 0.5, 1.0, 0.75, 0.5]
    },
    "pseudocode_based": {
      "correct": 4,
      "total": 5,
      "accuracy": 0.8,
      "avg_pass_rate": 0.85,
      "pass_rates": [1.0, 0.75, 1.0, 1.0, 0.5]
    },
    "details": [...]
  }
]
```

## Datasets

### HumanEval
- **Source**: OpenAI HumanEval benchmark
- **Problems**: 164 Python programming problems
- **Format**: Function signature + docstring + test cases

### MBPP
- **Source**: Mostly Basic Programming Problems
- **Problems**: 974 Python programming problems
- **Format**: Natural language description + test assertions

## How It Works

### Hint-Based Approach

1. **Question + Answer → Qwen2.5 3B**
   ```
   Input: "Write a function to check if two numbers are close"
   Output: "Hint 1: Iterate through pairs of numbers
            Hint 2: Calculate absolute difference
            Hint 3: Compare difference with threshold"
   ```

2. **Question + Hints → Qwen2.5 Coder 3B**
   ```
   Input: Question + Hints
   Output: Complete Python implementation
   ```

### Pseudocode-Based Approach

1. **Question + Answer → Qwen2.5 3B**
   ```
   Input: "Write a function to check if two numbers are close"
   Output: "
   FOR each element in list:
       FOR each other element:
           IF distance < threshold:
               RETURN true
   RETURN false"
   ```

2. **Question + Pseudocode → Qwen2.5 Coder 3B**
   ```
   Input: Question + Pseudocode
   Output: Complete Python implementation
   ```

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart the container
docker-compose restart
```

### Memory Issues

Increase Docker memory limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 16G  # Increase this
```

### Model Download Failures

```bash
# Manually pull models
docker-compose exec llm-evaluator ollama pull qwen2.5:3b
docker-compose exec llm-evaluator ollama pull qwen2.5-coder:3b
```

### Dataset Loading Issues

The system includes mock datasets as fallback. If datasets fail to load, check internet connection and HuggingFace availability.

## Advanced Usage

### Run Specific Dataset Only

Modify `main()` function in `evaluator.py`:

```python
# Only run HumanEval
humaneval_results = evaluator.run_evaluation("HumanEval", n_samples=10)
evaluator.print_summary(humaneval_results)
```

### Add Custom Test Cases

Extend the mock dataset methods:

```python
def get_custom_problems(self) -> List[Dict]:
    return [
        {
            "task_id": "custom_1",
            "question": "Your problem description",
            "answer": "Your solution code",
            "test_cases": [{"code": "assert function(input) == expected"}]
        }
    ]
```

### Export Results to CSV

Add to `main()` function:

```python
import pandas as pd

# Convert to DataFrame and export
df = pd.DataFrame(results['details'])
df.to_csv('/app/results/detailed_results.csv', index=False)
```

## Cleanup

Stop and remove containers:

```bash
docker-compose down

# Remove volumes (including downloaded models)
docker-compose down -v
```

## Performance Expectations

- **Model Download**: 5-10 minutes (one-time)
- **Per Sample Evaluation**: 30-60 seconds
- **5 Samples (each dataset)**: ~10-15 minutes total
- **Full Evaluation (164 HumanEval + 100 MBPP)**: Several hours

## Citation

If you use this evaluation system in your research, please cite:

- HumanEval: Chen et al. "Evaluating Large Language Models Trained on Code"
- MBPP: Austin et al. "Program Synthesis with Large Language Models"
- Qwen2.5: Alibaba Cloud

## License

MIT License - Feel free to modify and distribute.

## Contributing

Contributions welcome! Areas for improvement:
- Support for more model families
- Additional evaluation metrics
- Better error handling
- Visualization of results
- Support for other programming languages