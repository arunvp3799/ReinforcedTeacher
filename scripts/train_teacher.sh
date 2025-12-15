#!/bin/bash
#
# RLT-Style Hint Generation Training Script
#
# This script trains a teacher model (Qwen2.5-3B-Instruct) to generate helpful
# hints for coding problems using GRPO (Group Relative Policy Optimization).
#
# Hardware Requirements:
# - 2 GPUs (tested with A100 40GB or similar)
# - GPU 0: Teacher model training and rollout
# - GPU 1: Student model for reward computation (loaded automatically)
#
# Usage:
#   ./scripts/train_teacher.sh [additional_args]
#
# Example:
#   ./scripts/train_teacher.sh trainer.total_epochs=5
#

set -x  # Print commands as they execute

# =============================================================================
# Configuration
# =============================================================================

# Project paths
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATA_DIR="${PROJECT_DIR}/data/apps_hints"
export OUTPUT_DIR="${PROJECT_DIR}/outputs/teacher_grpo"

# Create output directory
mkdir -p "${OUTPUT_DIR}/checkpoints"
mkdir -p "${OUTPUT_DIR}/logs"

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1

# Ensure we use the right GPU for verl (teacher)
# Student model will be loaded on cuda:1 by the reward function
export VERL_CUDA_DEVICE=0

# Ray configuration for single node
export RAY_DEDUP_LOGS=0

# Disable ROCm (for AMD GPUs)
unset ROCR_VISIBLE_DEVICES

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo "========================================"
echo "RLT Hint Generation Training"
echo "========================================"
echo "Project directory: ${PROJECT_DIR}"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Check if data exists
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Error: Training data not found at ${DATA_DIR}/train.parquet"
    echo "Please run the data preparation script first:"
    echo "  python src/data/prepare_apps.py --output_dir ${DATA_DIR}"
    exit 1
fi

# Check GPU availability
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# =============================================================================
# Training
# =============================================================================

# Add project to Python path
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/verl:${PYTHONPATH}"

# Run training with verl
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.train_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    \
    custom_reward_function.path="${PROJECT_DIR}/src/rewards/rlt_reward.py" \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.student_model_name="Qwen/Qwen2.5-Coder-3B-Instruct" \
    +custom_reward_function.reward_kwargs.student_device="auto" \
    +custom_reward_function.reward_kwargs.alpha=0.1 \
    +custom_reward_function.reward_kwargs.length_penalty_threshold=10 \
    +custom_reward_function.reward_kwargs.max_hint_length=512 \
    +custom_reward_function.reward_kwargs.solution_copy_penalty=-5.0 \
    +custom_reward_function.reward_kwargs.empty_hint_penalty=-10.0 \
    +custom_reward_function.reward_kwargs.log_every_n=10 \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=10 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rlt_hint_generation' \
    trainer.experiment_name='teacher_grpo_apps' \
    trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
    "$@"

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Checkpoints saved to: ${OUTPUT_DIR}/checkpoints"
