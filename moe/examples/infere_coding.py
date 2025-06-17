import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from moe import Expert

# CLI argument for checkpoint path
parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=os.path.join("checkpoints", "expert_coding_pretrained.pt"),
    help="Path to the pretrained coding expert weights"
)
args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(device)

# Load and integrate coding expert into the first transformer block's MLP
expert = Expert(2048, 8192, 2048).to(device)
expert.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
model.transformer.h[0].mlp = expert
print("Coding expert loaded and integrated into GPT-Neo.")

# Sample prompts for testing
prompts = ["def sum(a, b):"]

# Generate with coding expert
print("\nTesting with coding expert:")
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=50,
        temperature=0.7,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    print(f"Prompt: {prompt}")
    print("Expert Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("Raw Output IDs:", outputs[0].tolist()[:10])  # Debug: first 10 token IDs