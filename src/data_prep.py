"""
Loads and prepares the Rotten Tomatoes sentiment dataset for training.
Run: python src/data_prep.py
"""

from datasets import load_dataset
import pandas as pd
import os

def load_and_prepare_data(save_dir="data"):
    print("Loading rotten_tomatoes dataset from Hugging Face...")
    dataset = load_dataset("rotten_tomatoes")

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")


    print("\nLabel distribution (train):")
    print(train_df["label"].value_counts())
    print("\nSample row:")
    print(train_df.iloc[0])

   
    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(f"{save_dir}/train.csv", index=False)
    val_df.to_csv(f"{save_dir}/val.csv", index=False)
    test_df.to_csv(f"{save_dir}/test.csv", index=False)

    print(f"\nSaved train/val/test CSVs to {save_dir}/")
    return train_df, val_df, test_df


if __name__ == "__main__":
    load_and_prepare_data()