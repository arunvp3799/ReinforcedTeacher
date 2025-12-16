"""
Quick test script to verify the setup before running full analysis.
Tests model loading and basic generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_dependencies():
    """Test if all required packages are installed."""
    print("Testing dependencies...")
    try:
        import torch
        import transformers
        import datasets
        print("✓ All required packages are installed")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - Transformers version: {transformers.__version__}")
        print(f"  - Datasets version: {datasets.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements.txt")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return "cuda"
    else:
        print("⚠ CUDA not available, will use CPU (slower)")
        return "cpu"


def test_model_loading(model_name="Qwen/Qwen2.5-Coder-3B-Instruct", device="cpu"):
    """Test loading a model."""
    print(f"\nTesting model loading: {model_name}")
    try:
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("  ✓ Tokenizer loaded")

        print("  Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        if device == "cpu":
            model = model.to(device)
        print("  ✓ Model loaded")

        # Test generation
        print("  Testing generation...")
        test_prompt = "def hello_world():"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  ✓ Generation successful")
        print(f"    Input: {test_prompt}")
        print(f"    Output: {generated_text}")

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_apps_dataset():
    """Test loading APPS dataset."""
    print("\nTesting APPS dataset loading...")
    try:
        from datasets import load_dataset
        print("  Loading dataset (this may take a moment)...")
        dataset = load_dataset("codeparrot/apps", split="train", streaming=True, trust_remote_code=True)

        # Get first example
        first_example = next(iter(dataset))
        print("  ✓ Dataset loaded successfully")
        print(f"    Sample problem ID: {first_example.get('problem_id', 'N/A')}")
        print(f"    Question length: {len(first_example['question'])} chars")
        return True
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("APPS Model Analysis - Setup Test")
    print("="*80)

    all_passed = True

    # Test dependencies
    if not test_dependencies():
        all_passed = False
        return

    # Test CUDA
    device = test_cuda()

    # Test model loading
    if not test_model_loading(device=device):
        all_passed = False

    # Test dataset
    if not test_apps_dataset():
        all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✓ All tests passed! You're ready to run the analysis.")
        print("\nNext steps:")
        print("  1. Run with default settings:")
        print("     python run_analysis.py")
        print("\n  2. Or run with custom settings:")
        print("     python run_analysis.py --num-examples 3 --device cuda")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
    print("="*80)


if __name__ == "__main__":
    main()
