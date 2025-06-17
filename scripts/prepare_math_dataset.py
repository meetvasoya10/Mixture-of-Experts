from datasets import load_dataset
import os

def filter_and_rename(dataset, column: str):
    """
    Keeps only the specified column and renames it to 'text' if needed.
    """
    if column != "text":
        dataset = dataset.rename_column(column, "text")
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
    return dataset


def prepare_math_dataset(save_dir="datasets", dataset_name="math_dataset", sample_size=200000):
    print("Preparing math dataset...")
    math = load_dataset("URSA-MATH/MMathCoT-1M", split=f"train[:{sample_size}]")
    math = filter_and_rename(math, "output")

    output_path = os.path.join(save_dir, dataset_name)
    math.save_to_disk(output_path)
    print(f"Math dataset saved to '{output_path}' with {len(math)} rows.")


if __name__ == "__main__":
    prepare_math_dataset()