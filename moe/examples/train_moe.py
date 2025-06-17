import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from moe import Expert

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "..", "..", "datasets"),
    help="Path to the datasets directory"
)
args = parser.parse_args()

# Device and precision setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()
print(f"Using device: {device}")

# Tokenizer and GPT-Neo (for embedding and lm_head reuse)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token
gpt_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
embedding = gpt_model.transformer.wte
lm_head = gpt_model.lm_head


def pretrain_expert(expert, dataset, task_name, epochs=5, batch_size=2, max_length=2048):
    expert.to(device)
    embedding.to(device)
    lm_head.to(device)
    optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-5, weight_decay=1e-4)

    # Train-val split
    split = dataset.train_test_split(test_size=0.1)
    train_dataset = split["train"]
    val_dataset = split["test"]
    print(f"{task_name} train size: {len(train_dataset)}, val size: {len(val_dataset)}")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding=False)

    # Tokenization and formatting
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=["text"])
    train_dataset.set_format("torch", columns=["input_ids"])
    val_dataset.set_format("torch", columns=["input_ids"])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(
            [d["input_ids"] for d in x], batch_first=True, padding_value=tokenizer.eos_token_id
        ),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(
            [d["input_ids"] for d in x], batch_first=True, padding_value=tokenizer.eos_token_id
        ),
    )

    best_val_loss = float("inf")
    for epoch in range(epochs):
        expert.train()
        total_train_loss = 0

        for i, batch in enumerate(train_loader):
            inputs = batch.to(device)
            labels = inputs.clone()

            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                embedded = embedding(inputs)
                outputs = expert(embedded)
                logits = lm_head(outputs)
                loss = nn.CrossEntropyLoss()(logits.view(-1, tokenizer.vocab_size), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_train_loss += loss.item()
            if i % 100 == 0:
                print(f"{task_name} Epoch {epoch+1}, Batch {i}, Train Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        expert.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch.to(device)
                labels = inputs.clone()

                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    embedded = embedding(inputs)
                    outputs = expert(embedded)
                    logits = lm_head(outputs)
                    loss = nn.CrossEntropyLoss()(logits.view(-1, tokenizer.vocab_size), labels.view(-1))

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"{task_name} Epoch {epoch+1}, Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_dir = "./checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(expert.state_dict(), os.path.join(checkpoint_dir, f"expert_{task_name}_pretrained.pt"))
            print(f"Saved best model for {task_name} with Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    print("Loading datasets for pretraining...")
    coding = load_from_disk(os.path.join(args.data_path, "merged_coding_200k"))
    cot = load_from_disk(os.path.join(args.data_path, "cot_dataset"))
    math = load_from_disk(os.path.join(args.data_path, "math_dataset"))

    print("Pretraining experts...")
    pretrain_expert(Expert(2048, 8192, 2048), coding, "coding")
    pretrain_expert(Expert(2048, 8192, 2048), cot, "cot")
    pretrain_expert(Expert(2048, 8192, 2048), math, "math")

    print("Training complete! Models saved in /checkpoints directory.")