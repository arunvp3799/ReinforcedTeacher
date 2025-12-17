import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_PATH = "checkpoints/sft_qwen2.5_coder_3b_merged"
OUTPUT_FILE = "model_outputs.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(path):
    print(f"Loading model from {path}...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto"
    )
    return model, tokenizer

def generate_completion(model, tokenizer, messages, max_new_tokens=512):
    """Generates using ChatML template to match SFT training."""
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    full_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return full_text.strip()

def main():
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    all_results = []
    
    system_prompt = "You are an intelligent coding assistant. Given a programming problem, write a correct and efficient Python solution."

    # --- 1. PROCESS HUMANEVAL ---
    print("\nGenerating HumanEval...")
    he_dataset = load_dataset("openai_humaneval", split="test")
    for ex in tqdm(he_dataset):
        user_content = f"Complete the following Python function:\n\n{ex['prompt']}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        completion = generate_completion(model, tokenizer, messages)
        
        all_results.append({
            "task_id": ex["task_id"],
            "prompt": ex["prompt"],
            "completion": completion,
            "test": ex["test"],
            "entry_point": ex["entry_point"]
        })

    # --- 2. PROCESS MBPP ---
    print("\nGenerating MBPP...")
    mbpp_dataset = load_dataset("mbpp", split="test")
    for ex in tqdm(mbpp_dataset):
        # We pass the instruction. The Docker evaluator will handle name fixing.
        user_content = f"Requirement: {ex['text']}\n\nWrite a complete Python function."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        completion = generate_completion(model, tokenizer, messages)
        
        all_results.append({
            "task_id": f"MBPP/{ex['task_id']}",
            "prompt": "", # Not needed for MBPP in the docker evaluator
            "completion": completion,
            "test_list": ex["test_list"],
            "test_setup_code": "" # Standard MBPP doesn't usually need setup
        })

    # --- SAVE RESULTS ---
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDone! Generation saved to {OUTPUT_FILE}")
    print("You can now run the Docker evaluator on this file.")

if __name__ == "__main__":
    main()
