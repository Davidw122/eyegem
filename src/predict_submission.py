import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_and_verify_model(checkpoint_path):
    """Loads the model and tokenizer from a local directory from a cold start."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"CRITICAL ERROR: Checkpoint directory '{checkpoint_path}' not found.")
        sys.exit(1)

    try:
        # device_map="auto" ensures it finds the GPU if available, or falls back gracefully
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            torch_dtype=torch.float16 # Standard for hackathon inference speed
        )
        print("Model and tokenizer successfully loaded into memory.")
        return model, tokenizer
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model weights. Did they save correctly?\n{e}")
        sys.exit(1)

def run_test_inference(model, tokenizer, prompt):
    """Runs a single prediction to prove the weights are active and generating."""
    print(f"\n--- Testing Inference ---")
    print(f"Input Prompt: {prompt}")
    
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50,
            temperature=0.0 # Greedy decoding for deterministic testing
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model Output: {response}")
    print(f"--- Test Complete ---\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cold-start checkpoint verification.")
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True, 
        help="Relative path to the saved model (e.g., checkpoints/pilot_run_1)"
    )
    parser.add_argument(
        "--test_prompt", 
        type=str, 
        default="Question: What is the capital of France? Answer:", 
        help="A simple string to test if the model generates text."
    )
    
    args = parser.parse_args()
    
    # 1. Verify we are running from the root directory
    if not os.path.exists("src") or not os.path.exists("checkpoints"):
        print("WARNING: Please run this script from the root ic3_project/ directory.")
    
    # 2. Execute cold start
    model, tokenizer = load_and_verify_model(args.checkpoint_dir)
    run_test_inference(model, tokenizer, args.test_prompt)