from datasets import load_dataset
import os

def filter_and_rename(dataset, column: str):
    """
    Keep only the specified column and rename it to 'text'.
    """
    dataset = dataset.rename_column(column, "text")
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
    return dataset


def prepare_cot_dataset(save_dir="datasets", dataset_name="cot_dataset", sample_size=200000):
    print("Preparing CoT dataset...")
    cot = load_dataset("meta-math/MetaMathQA", split=f"train[:{sample_size}]")
    cot = filter_and_rename(cot, "response")

    output_path = os.path.join(save_dir, dataset_name)
    cot.save_to_disk(output_path)
    print(f"CoT dataset saved to '{output_path}' with {len(cot)} rows.")


if __name__ == "__main__":
    prepare_cot_dataset()