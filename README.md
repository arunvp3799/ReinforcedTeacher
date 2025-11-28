# ReinforcedTeacher

## Setup

```bash
pip install -r requirements.txt
```

## Run Evaluation

Single dataset:
```bash
python src/evaluate/vllm_evaluate.py --model_name <model_name> --data <humaneval|mbpp> --batch_size 16 --output_folder results --num_workers 4 --max_model_len 2048
```

Both datasets:
```bash
bash src/evaluate/run_evaluation.sh <model_name> [batch_size] [output_folder] [num_workers] [max_model_len]
```
