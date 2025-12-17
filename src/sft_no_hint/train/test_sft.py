import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import re

def load_model(model_path):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available! Model will run on CPU (very slow)")
        print("To enable GPU in Colab: Runtime → Change runtime type → GPU")
        device = "cpu"
        torch_dtype = torch.float32
    else:
        print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda:0"
        torch_dtype = torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        use_fast=True,
    )
    
    # Load model directly to specified device (not using device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Explicitly move to device
    if device != "cpu":
        print(f"Moving model to {device}...")
        model = model.to(device)
    
    # Verify device
    actual_device = next(model.parameters()).device
    print(f"✓ Model loaded on device: {actual_device}")
    
    if actual_device.type == "cpu" and torch.cuda.is_available():
        print("\n⚠️  WARNING: GPU is available but model is on CPU!")
        print("This shouldn't happen. Trying to force move to GPU...")
        try:
            model = model.cuda()
            print(f"✓ Successfully moved to: {next(model.parameters()).device}")
        except Exception as e:
            print(f"Failed to move to GPU: {e}")
            response = input("Continue on CPU anyway? (y/n): ")
            if response.lower() != 'y':
                exit()
    
    return model, tokenizer

def generate_code(model, tokenizer, prompt, max_new_tokens=512, temperature=0.2):
    """Generate code completion for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the prompt)
    if prompt in generated_text:
        generated_code = generated_text[len(prompt):].strip()
    else:
        generated_code = generated_text.strip()
    
    return generated_code

def generate_code_batch(model, tokenizer, prompts, max_new_tokens=512, temperature=0.2, batch_size=8):
    """Generate code for multiple prompts at once (faster)."""
    all_generated = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        for j, output in enumerate(outputs):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            prompt = batch_prompts[j]
            
            # Extract only the generated part
            if prompt in generated_text:
                generated_code = generated_text[len(prompt):].strip()
            else:
                generated_code = generated_text.strip()
            
            all_generated.append(generated_code)
        
        # Clear memory after each batch
        del inputs, outputs
        torch.cuda.empty_cache()
    
    return all_generated

def extract_code_from_generation(generated_text, entry_point=None):
    """Extract clean Python code from model generation."""
    # Try to extract code from markdown code blocks
    code_block_pattern = r"```python\n(.*?)\n```"
    matches = re.findall(code_block_pattern, generated_text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # Try to extract code without markdown
    code_block_pattern = r"```\n(.*?)\n```"
    matches = re.findall(code_block_pattern, generated_text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # If no code blocks, try to find function definition
    if entry_point:
        # Find everything from "def entry_point" to the end or next "def"
        pattern = rf"(def {entry_point}.*?)(?=\ndef |\Z)"
        matches = re.findall(pattern, generated_text, re.DOTALL)
        if matches:
            return matches[0].strip()
    
    # Return the whole text if nothing else works
    return generated_text.strip()


def generate_code_chat(model, tokenizer, messages, max_new_tokens=512, temperature=0.2):
    """
    Generate code using the ChatML format used during SFT.
    """
    # 1. Apply the chat template to match training format
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
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # 2. Decode and extract only the new tokens
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

def evaluate_humaneval(model, tokenizer, output_file="humaneval_results.jsonl", max_samples=None):
    """Evaluate model on HumanEval dataset using ChatML."""
    print("\n" + "="*50)
    print("Evaluating on HumanEval (Chat Format)")
    print("="*50)
    
    dataset = load_dataset("openai_humaneval", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Match the system prompt from File 1
    system_prompt = "You are an intelligent coding assistant. Given a programming problem, write a correct and efficient Python solution."
    results = []

    for example in tqdm(dataset, desc="Processing HumanEval"):
        # Format the user prompt exactly as in training
        user_content = f"Complete the following Python function:\n\n{example['prompt']}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Generate
        clean_code = generate_code_chat(model, tokenizer, messages)
        clean_code = extract_code_from_generation(clean_code, example["entry_point"])
        
        # Combine for execution
        full_code = example["prompt"] + clean_code
        
        results.append({
            "task_id": example["task_id"],
            "prompt": example["prompt"],
            "generated_code": clean_code,
            "full_code": full_code,
            "test": example["test"],
            "entry_point": example["entry_point"]
        })

    with open(output_file, "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    return results

def evaluate_mbpp(model, tokenizer, output_file="mbpp_results.jsonl", max_samples=None):
    """Evaluate model on MBPP dataset using ChatML."""
    print("\n" + "="*50)
    print("Evaluating on MBPP (Chat Format)")
    print("="*50)
    
    dataset = load_dataset("mbpp", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    system_prompt = "You are an intelligent coding assistant. Given a programming problem, write a correct and efficient Python solution."
    results = []

    for example in tqdm(dataset, desc="Processing MBPP"):
        user_content = f"Write a Python function that satisfies the following requirement:\n\n{example['text']}\n\nProvide a complete, working Python function."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        clean_code = generate_code_chat(model, tokenizer, messages)
        clean_code = extract_code_from_generation(clean_code)
        
        results.append({
            "task_id": example["task_id"],
            "text": example["text"],
            "generated_code": clean_code,
            "test_list": example["test_list"]
        })

    with open(output_file, "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    return results



def run_tests_humaneval(results_file="humaneval_results.jsonl"):
    """Execute tests for HumanEval results."""
    print("\n" + "="*50)
    print("Running HumanEval Tests")
    print("="*50)
    
    with open(results_file, "r") as f:
        results = [json.loads(line) for line in f]
    
    passed = 0
    total = 0
    
    for result in tqdm(results, desc="Testing HumanEval"):
        task_id = result["task_id"]
        full_code = result["full_code"]
        test = result["test"]
        entry_point = result["entry_point"]
        
        # Create test code
        test_code = full_code + "\n\n" + test + f"\n\ncheck({entry_point})"
        
        try:
            # Execute the test
            exec(test_code, {})
            passed += 1
            result["passed"] = True
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
        
        total += 1
    
    # Save results with pass/fail
    output_file = results_file.replace(".jsonl", "_tested.jsonl")
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    pass_rate = (passed / total) * 100 if total > 0 else 0
    print(f"\nHumanEval Results:")
    print(f"Passed: {passed}/{total} ({pass_rate:.2f}%)")
    print(f"Detailed results saved to {output_file}")
    
    return passed, total

def run_tests_mbpp(results_file="mbpp_results.jsonl"):
    """Execute tests for MBPP results."""
    print("\n" + "="*50)
    print("Running MBPP Tests")
    print("="*50)
    
    with open(results_file, "r") as f:
        results = [json.loads(line) for line in f]
    
    passed = 0
    total = 0
    
    for result in tqdm(results, desc="Testing MBPP"):
        task_id = result["task_id"]
        generated_code = result["generated_code"]
        test_list = result["test_list"]
        
        all_tests_passed = True
        errors = []
        
        for test in test_list:
            # Create test code
            test_code = generated_code + "\n\n" + test
            
            try:
                # Execute the test
                exec(test_code, {})
            except Exception as e:
                all_tests_passed = False
                errors.append(str(e))
        
        if all_tests_passed:
            passed += 1
            result["passed"] = True
        else:
            result["passed"] = False
            result["errors"] = errors
        
        total += 1
    
    # Save results with pass/fail
    output_file = results_file.replace(".jsonl", "_tested.jsonl")
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    pass_rate = (passed / total) * 100 if total > 0 else 0
    print(f"\nMBPP Results:")
    print(f"Passed: {passed}/{total} ({pass_rate:.2f}%)")
    print(f"Detailed results saved to {output_file}")
    
    return passed, total

def main():
    # Configuration
    MODEL_PATH = "checkpoints/sft_qwen2.5_coder_3b_merged"  # Path to your merged model
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please provide the correct path to your fine-tuned model.")
        return
    
    # Load model
    model, tokenizer = load_model(MODEL_PATH)
    
    # Evaluate on HumanEval
    humaneval_results = evaluate_humaneval(model, tokenizer)
    
    # Evaluate on MBPP
    mbpp_results = evaluate_mbpp(model, tokenizer, max_samples=100)  # Test 100 samples with batching
    
    # Run tests
    print("\n" + "="*50)
    print("Running Tests")
    print("="*50)
    
    he_passed, he_total = run_tests_humaneval("humaneval_results.jsonl")
    mbpp_passed, mbpp_total = run_tests_mbpp("mbpp_results.jsonl")
    
    # Print final summary
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"HumanEval: {he_passed}/{he_total} ({(he_passed/he_total)*100:.2f}%)")
    print(f"MBPP: {mbpp_passed}/{mbpp_total} ({(mbpp_passed/mbpp_total)*100:.2f}%)")

if __name__ == "__main__":
    main()
