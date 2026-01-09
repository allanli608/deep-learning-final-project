import pandas as pd
import numpy as np
import os
import argparse
import Levenshtein  # You will need to install this: pip install python-Levenshtein
from sklearn.model_selection import train_test_split


def load_wnc(file_path):
    """
    Loads the WNC biased.full dataset.
    The schema is based on the Pryzant et al. repo standards.
    """
    print(f"Loading raw data from {file_path}...")

    # WNC 'biased.full' usually has no headers.
    # Columns: id, src_tok, tgt_tok, src_raw, tgt_raw, pos_tags, parse_tree
    column_names = ["id", "src_tok", "tgt_tok", "src_raw", "tgt_raw", "pos", "parse"]

    try:
        # on_bad_lines='skip' handles potential formatting errors in the raw TSV
        df = pd.read_csv(
            file_path, sep="\t", names=column_names, on_bad_lines="skip", quoting=3
        )
        print(f"Loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def filter_structural_bias(df, min_diff=2, max_diff_ratio=0.4):
    """
    Filters for 'Structural/Multi-word' bias.

    Args:
        min_diff: Minimum number of word changes required (default 2).
                  Filters out simple single-word swaps.
        max_diff_ratio: Max changes allowed relative to sentence length (default 0.4).
                        Filters out total rewrites/hallucinations/vandalism fixes.
    """
    print("Filtering for structural bias (Deep Learning 'Hard Mode')...")

    filtered_rows = []

    for idx, row in df.iterrows():
        src = str(row["src_raw"])
        tgt = str(row["tgt_raw"])

        # Calculate word-level Levenshtein distance
        src_words = src.split()
        tgt_words = tgt.split()

        # Quick sanity check for empty strings
        if len(src_words) == 0 or len(tgt_words) == 0:
            continue

        # Distance: Number of insertions, deletions, or substitutions
        dist = Levenshtein.distance(src_words, tgt_words)

        # Ratio: How much of the sentence changed?
        # If dist is 10 and sentence is 20 words, ratio is 0.5
        ratio = dist / max(len(src_words), len(tgt_words))

        # LOGIC:
        # 1. dist >= min_diff: We want multi-word edits (Structural Bias)
        # 2. ratio <= max_diff_ratio: We want to KEEP most of the content (Meaning Preservation)
        if dist >= min_diff and ratio <= max_diff_ratio:
            filtered_rows.append(row)

    filtered_df = pd.DataFrame(filtered_rows)
    print(
        f"Filtered down to {len(filtered_df)} complex examples (from {len(df)} original)."
    )
    return filtered_df


def save_splits(df, output_dir):
    """
    Splits into Train/Val/Test and saves as CSV for HuggingFace.
    """
    print("Splitting datasets...")

    # 80% Train, 10% Val, 10% Test
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    os.makedirs(output_dir, exist_ok=True)

    train.to_csv(os.path.join(output_dir, "train_complex.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "val_complex.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test_complex.csv"), index=False)

    print(f"Saved to {output_dir}:")
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/WNC/biased.full")
    parser.add_argument("--output_dir", type=str, default="data/processed/")
    args = parser.parse_args()

    # Run pipeline
    df = load_wnc(args.input_file)
    if df is not None:
        df_complex = filter_structural_bias(df)
        save_splits(df_complex, args.output_dir)
