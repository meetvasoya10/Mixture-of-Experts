import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, concatenate_datasets
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from moe import MoE, Expert

# Argument parser for flexibility
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "..", "..", "datasets"),
    help="Base path to datasets directory",
)
args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("CUDA memory cleared!")
    print(f"Initial memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token


def finetune_integrated_moe(model, dataset, epochs=2, batch_size=1, max_length=2048):
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.transformer.h[0].mlp.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.transformer.h[0].mlp.parameters(), lr=1e-5)

    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=max_length, padding=False
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch", columns=["input_ids"])

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(
            [d["input_ids"] for d in x],
            batch_first=True,
            padding_value=tokenizer.eos_token_id,
        ),
    )

    print(f"Finetuning MoE, dataset size: {len(dataset)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            inputs = batch.to(device)
            labels = inputs.clone()

            outputs = model(inputs, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Finetuning Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        print(f"Finetuning Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(os.path.join(checkpoint_dir, "moe_gpt_neo_1.3b"))
    tokenizer.save_pretrained(os.path.join(checkpoint_dir, "moe_gpt_neo_1.3b"))
    print("Fine-tuning complete! Model saved at ./checkpoints/moe_gpt_neo_1.3b")


if __name__ == "__main__":
    print("Loading datasets for fine-tuning...")
    coding = load_from_disk(os.path.join(args.data_path, "merged_coding_200k"))
    cot = load_from_disk(os.path.join(args.data_path, "cot_dataset"))
    math = load_from_disk(os.path.join(args.data_path, "math_dataset"))

    print("Preparing mixed dataset for gating fine-tuning...")
    mixed_coding = coding.select(range(30000)).train_test_split(test_size=0.1)["train"]
    mixed_cot = cot.select(range(30000)).train_test_split(test_size=0.1)["train"]
    mixed_math = math.select(range(30000)).train_test_split(test_size=0.1)["train"]
    mixed_dataset = concatenate_datasets([mixed_coding, mixed_cot, mixed_math])
    print(f"Mixed dataset ready: {len(mixed_dataset)} rows")

    print("Building MoE with pretrained experts...")
    moe = MoE(hidden_size=2048, num_experts=3, expert_hidden_dim=8192)
    checkpoint_dir = "./checkpoints"
    moe.experts[0].load_state_dict(torch.load(os.path.join(checkpoint_dir, "expert_coding_pretrained.pt")))
    moe.experts[1].load_state_dict(torch.load(os.path.join(checkpoint_dir, "expert_cot_pretrained.pt")))
    moe.experts[2].load_state_dict(torch.load(os.path.join(checkpoint_dir, "expert_math_pretrained.pt")))

    print("Loading GPT-Neo-1.3B and integrating MoE...")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model.transformer.h[0].mlp = moe

    print("Fine-tuning integrated MoE...")
    finetune_integrated_moe(model, mixed_dataset)