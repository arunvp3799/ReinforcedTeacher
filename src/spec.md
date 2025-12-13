# RLT-Style Hint Generation for Code: Project Specification

## Executive Summary

This project implements a **Reinforcement Learning Teacher (RLT)** framework adapted for code generation. The goal is to train a teacher model to generate helpful hints that improve a student model's ability to solve coding problems — without the teacher needing to solve problems itself.

---

## Problem Statement

### Core Hypothesis

We want to prove that:

```
S(with RL-trained hint) > S(with base hint) > S(without hint)
```

Where `S` represents a student model's pass rate on coding benchmarks.

### The Challenge

Traditional approaches to improving code generation either:
1. **Fine-tune the model directly** — expensive, requires large compute
2. **Use standard RL** — requires the model to explore and solve problems, facing sparse rewards
3. **Distill from stronger models** — depends on having access to much larger/better models

We want a method where a **small teacher model** can learn to provide hints that help a student model solve problems, even if the teacher itself cannot solve those problems.

---

## Background: Related Work

### Paper 1: WST (Weak-to-Strong Transfer)
- Small teacher generates instructions for a larger student
- Teacher is trained with RL based on student's downstream performance
- Focus: General prompt engineering for reasoning tasks

### Paper 2: RLT (Reinforcement-Learned Teachers)
- Key insight: Teachers don't need to solve problems — they need to **explain** solutions
- Teacher sees both the question AND the solution, generates explanations
- Reward: How well the student understands the solution given the explanation
- Measured via student's log-probability of the correct answer

### Gap We Address
Neither paper tackles **code generation**. Code has unique properties:
- Verifiable correctness via test cases
- Structured output with syntax constraints
- Multiple valid solutions possible
- Rich intermediate feedback available

---

## Proposed Solution

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐     ┌──────────────┐                    │
│   │   Question   │     │   Solution   │                    │
│   │  (from APPS) │     │ (correct code)│                    │
│   └──────┬───────┘     └──────┬───────┘                    │
│          │                    │                             │
│          └────────┬───────────┘                             │
│                   ▼                                         │
│          ┌────────────────┐                                 │
│          │ Teacher Model  │  (Qwen2.5-3B-Instruct)         │
│          │  Being Trained │                                 │
│          └────────┬───────┘                                 │
│                   │                                         │
│                   ▼                                         │
│              ┌─────────┐                                    │
│              │  Hint   │                                    │
│              └────┬────┘                                    │
│                   │                                         │
│          ┌────────┴────────┐                                │
│          │                 │                                │
│          ▼                 ▼                                │
│   ┌─────────────┐   ┌─────────────┐                        │
│   │  Question   │   │   Hint      │                        │
│   └──────┬──────┘   └──────┬──────┘                        │
│          │                 │                                │
│          └────────┬────────┘                                │
│                   ▼                                         │
│          ┌────────────────┐                                 │
│          │ Student Model  │  (Qwen2.5-Coder-3B) [FROZEN]   │
│          └────────┬───────┘                                 │
│                   │                                         │
│                   ▼                                         │
│     ┌─────────────────────────────┐                        │
│     │ Compute: P(solution | q, h) │                        │
│     │    (log-probability)        │                        │
│     └──────────────┬──────────────┘                        │
│                    │                                        │
│                    ▼                                        │
│              ┌──────────┐                                   │
│              │  Reward  │ ──► Update Teacher via GRPO      │
│              └──────────┘                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Teacher Model
- **Model**: Qwen2.5-3B-Instruct
- **Input**: Coding problem + correct solution
- **Output**: A hint that guides toward the solution without giving it away
- **Training**: GRPO (Group Relative Policy Optimization)

#### 2. Student Model
- **Model**: Qwen2.5-Coder-3B-Instruct
- **Role**: Frozen during teacher training, used only for reward computation
- **Purpose**: Measures how helpful hints are

#### 3. Reward Function (RLT-Style)
The reward measures: "Given this hint, how likely is the student to produce the correct solution?"

```
reward = mean(log P_student(solution | question, hint)) 
       + α * min(log P_student(solution | question, hint))
```

The `min` term ensures no token is completely unexplained by the hint.

#### 4. RL Algorithm: GRPO
- No critic/value network needed
- For each problem, generate K hints (e.g., K=4)
- Compute rewards for each hint
- Normalize rewards within the group (relative advantage)
- Update teacher to favor higher-reward hints

---

## Data Pipeline

### Training Data: APPS Dataset
- Source: `codeparrot/apps` on HuggingFace
- Content: 10,000 coding problems with solutions and test cases
- Filtering: Use "introductory" and "interview" difficulty levels
- Format for teacher:
  ```
  Input: <problem> + <solution>
  Output: <hint>
  ```

### Evaluation Data
- **HumanEval**: 164 Python programming problems (OpenAI)
- **MBPP**: 974 Python programming problems (Google)

---

## Evaluation Protocol

### Three Conditions to Compare

| Condition | Description |
|-----------|-------------|
| **No Hint** | Student solves problem with only the question |
| **Base Hint** | Student gets hint from base teacher (before RL) |
| **RL Hint** | Student gets hint from RL-trained teacher |

### Metrics
- **pass@1**: Does the first generated solution pass all tests?
- **pass@10**: Does any of 10 generated solutions pass?
- **pass@100**: Does any of 100 generated solutions pass?

### Expected Outcome
```
pass@k(RL Hint) > pass@k(Base Hint) > pass@k(No Hint)
```

---

## Technical Implementation Details

### Framework: verl
- Flexible RL training library from ByteDance
- Native support for GRPO algorithm
- Integrates with vLLM for fast generation
- Custom reward function support

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Teacher Model | Qwen2.5-3B-Instruct | Small but capable of generating coherent hints |
| Student Model | Qwen2.5-Coder-3B-Instruct | Code-specialized, frozen during training |
| RL Algorithm | GRPO | No critic needed, stable training |
| Rollouts per prompt | 4 | Balance between compute and variance reduction |
| Max hint length | 512 tokens | Hints should be concise |
| KL penalty | In loss (not reward) | Standard GRPO approach |
| Learning rate | 1e-6 | Conservative for stable training |
| Total epochs | 10 | Monitor for convergence |

### Reward Function Design

The reward has several components:

1. **Student Understanding (r_SS)**: Log-probability of correct solution given hint
2. **Interpretability (r_KL)**: Hint should make sense without seeing the solution
3. **Sanity Penalties**:
   - Penalize empty/trivial hints
   - Penalize hints that copy the solution directly
   - Bonus for appropriate hint length

---

## Directory Structure

```
rlt-code-hints/
├── configs/
│   └── teacher_grpo.yaml          # verl training configuration
├── data/
│   ├── prepare_apps.py            # Data preprocessing script
│   └── processed/                 # Output parquet files
├── rewards/
│   └── rlt_reward.py              # Custom RLT reward function
├── models/
│   └── student_inference.py       # Student model utilities
├── eval/
│   ├── eval_humaneval.py          # HumanEval evaluation
│   └── eval_mbpp.py               # MBPP evaluation
├── scripts/
│   └── train_teacher.sh           # Training launch script
├── checkpoints/                   # Saved model checkpoints
├── eval_results/                  # Evaluation outputs
├── requirements.txt
└── README.md
```

---

## Implementation Tasks

### Phase 1: Setup
1. Set up environment with verl, vLLM, transformers
2. Download and preprocess APPS dataset
3. Verify teacher and student models load correctly

### Phase 2: Reward Function
1. Implement RLT-style reward computation
2. Load frozen student model for log-prob calculation
3. Add sanity checks (empty hint, solution copying)
4. Test reward function standalone before training

### Phase 3: Training Pipeline
1. Create verl configuration for GRPO
2. Set up teacher prompt format (question + solution → hint)
3. Configure rollout settings (temperature, num samples)
4. Implement checkpointing and logging

### Phase 4: Evaluation
1. Implement HumanEval evaluation script
2. Implement MBPP evaluation script
3. Support three conditions: no hint, base hint, RL hint
4. Generate comparison tables and visualizations

### Phase 5: Analysis
1. Compare pass rates across conditions
2. Analyze hint quality (length, specificity, helpfulness)
3. Study failure cases
4. Document findings

---

## Key Design Decisions

### Why RLT-style (teacher sees solution)?
- Avoids exploration problem — teacher doesn't need to solve problems
- Dense reward signal — log-prob gives gradient for every token
- Small models can be effective teachers

### Why log-prob reward instead of execution?
- Much cheaper — no code execution during training
- Faster iteration — forward pass only
- Differentiable signal — better for learning
- Execution-based reward used only for final evaluation

### Why GRPO instead of PPO?
- No critic network needed — simpler setup
- Group normalization handles baseline estimation
- Proven effective for reasoning tasks (DeepSeek-R1)

### Why freeze the student during training?
- Stable reward signal — same hint gives same reward
- Cleaner experimental setup
- Can co-train later as extension

---

## Potential Novelty Angles

This implementation follows RLT closely but applies it to code. For additional novelty, consider:

1. **Hierarchical hints**: Multiple levels of specificity
2. **Student-adaptive hints**: Condition on student capability
3. **Multi-turn hinting**: Iterative refinement based on student attempts
4. **Hint type selection**: Choose between algorithm hints, edge case hints, etc.
5. **Co-training**: Update student and teacher together

---

## Success Criteria

1. **Primary**: RL-trained teacher hints improve student pass rate over base teacher
2. **Secondary**: Improvement is consistent across difficulty levels
3. **Tertiary**: Hints are qualitatively meaningful (not gaming the reward)

---

## Dependencies

- Python 3.10+
- PyTorch 2.4+
- transformers 4.45+
- verl 0.3+
- vLLM 0.6+
- datasets (HuggingFace)
- wandb (logging)

---

## Notes for Implementation

1. **Memory management**: Student model should be loaded once and reused across reward computations
2. **Tokenization alignment**: Ensure teacher and student tokenizers handle the same prompt format
3. **Hint extraction**: Teacher output needs parsing to extract just the hint content
4. **Test case format**: APPS has varying formats — handle both stdin/stdout and function-based problems
5. **Evaluation sandbox**: Use isolated execution environment for safety during evaluation