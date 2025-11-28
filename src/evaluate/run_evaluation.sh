#!/bin/bash

MODEL_NAME=$1
BATCH_SIZE=${2:-16}
OUTPUT_FOLDER=${3:-results}
NUM_WORKERS=${4:-4}
MAX_MODEL_LEN=${5:-2048}

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <model_name> [batch_size] [output_folder] [num_workers] [max_model_len]"
    exit 1
fi

echo "Running evaluation for model: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Output folder: $OUTPUT_FOLDER"
echo "Number of workers: $NUM_WORKERS"
echo "Max model length: $MAX_MODEL_LEN"

echo "Evaluating on HumanEval..."
python vllm_evaluate.py \
    --model_name "$MODEL_NAME" \
    --data humaneval \
    --batch_size $BATCH_SIZE \
    --output_folder "$OUTPUT_FOLDER" \
    --num_workers $NUM_WORKERS \
    --max_model_len $MAX_MODEL_LEN

echo "Evaluating on MBPP..."
python vllm_evaluate.py \
    --model_name "$MODEL_NAME" \
    --data mbpp \
    --batch_size $BATCH_SIZE \
    --output_folder "$OUTPUT_FOLDER" \
    --num_workers $NUM_WORKERS \
    --max_model_len $MAX_MODEL_LEN

echo "All evaluations completed!"
