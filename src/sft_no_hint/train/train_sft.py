import torch

from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import LoraConfig

from trl import (
    SFTTrainer,
    SFTConfig,
)

def train():
    MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
    DATA_PATH = "data/apps_sft/train_sft.jsonl"
    OUTPUT_DIR = "checkpoints/sft_qwen2.5_coder_3b"
    
    BATCH_SIZE = 4
    GRAD_ACCUM = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading data from {DATA_PATH}...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": 0},  # ← Force everything to GPU 0
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for training
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    print(f"Model on: {next(model.parameters()).device}")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    
    print("Merging model...")
    model = trainer.model.merge_and_unload()
    merged_output = f"{OUTPUT_DIR}_merged"
    model.save_pretrained(merged_output)
    tokenizer.save_pretrained(merged_output)
    print(f"Merged model saved to {merged_output}")

if __name__ == "__main__":
    train()
