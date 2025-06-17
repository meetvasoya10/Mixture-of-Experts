from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
import os


def prepare_merged_coding_dataset(
    base_path="datasets",
    alpaca_repo="iamtarun/code_instructions_120k_alpaca",
    output_dir="merged_coding_200k"
):
    print("Loading existing coding dataset...")
    coding_dataset = load_from_disk(os.path.join(base_path, "coding_dataset"))
    print(f"Original coding size: {len(coding_dataset)} rows")

    print("Loading Alpaca code instructions (streaming)...")
    alpaca_stream = load_dataset(alpaca_repo, split="train", streaming=True)

    print("Converting Alpaca stream to text format...")
    alpaca_rows = []
    for example in alpaca_stream:
        if isinstance(example.get("output"), str):
            text = f"{example['instruction']} {example['input']} {example['output']}".strip()
            alpaca_rows.append({"text": text})
        if len(alpaca_rows) >= 100000:  # optional limit
            break

    print(f"Loaded {len(alpaca_rows)} Alpaca examples.")
    alpaca_dataset = Dataset.from_list(alpaca_rows)

    print("Merging datasets...")
    merged_dataset = concatenate_datasets([coding_dataset, alpaca_dataset])
    merged_dataset = merged_dataset.shuffle(seed=42)

    output_path = os.path.join(base_path, output_dir)
    print(f"Saving merged dataset: {len(merged_dataset)} rows → {output_path}")
    merged_dataset.save_to_disk(output_path)


if __name__ == "__main__":
    prepare_merged_coding_dataset()