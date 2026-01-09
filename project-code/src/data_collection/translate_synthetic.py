import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import argparse

# Configuration
MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"
BATCH_SIZE = 128


def load_translator(device):
    """Loads the lightweight English-to-Chinese translation model."""
    print(f"Loading translation model: {MODEL_NAME}...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

    # CHANGE 2: Use Half Precision (fp16) for massive speedup
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(device).half()

    model.eval()
    return tokenizer, model


def translate_batch(text_list, tokenizer, model, device):
    """Translates a list of strings from English to Chinese."""

    # tokenize
    inputs = tokenizer(
        text_list, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    # generate
    with torch.no_grad():
        translated = model.generate(
            **inputs,
            # CHANGE 3: Greedy Decoding (Fastest)
            num_beams=1,
            do_sample=False,
            max_new_tokens=512,  # Prevents runaway generation
        )

    # decode
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def process_file(input_path, output_path, tokenizer, model, device):
    print(f"\nProcessing {input_path}...")

    # Load Data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Skipping {input_path} (File not found)")
        return

    # Check columns
    if "src_raw" not in df.columns or "tgt_raw" not in df.columns:
        print(f"Skipping {input_path} (Missing columns 'src_raw' or 'tgt_raw')")
        return

    # Create new lists
    zh_src = []
    zh_tgt = []

    # Iterate in batches
    total_rows = len(df)
    for i in tqdm(range(0, total_rows, BATCH_SIZE), desc="Translating"):
        batch_slice = slice(i, i + BATCH_SIZE)

        # Get English batch
        batch_src_en = df["src_raw"].iloc[batch_slice].tolist()
        batch_tgt_en = df["tgt_raw"].iloc[batch_slice].tolist()

        # Translate Source (Biased)
        batch_src_zh = translate_batch(batch_src_en, tokenizer, model, device)

        # Translate Target (Neutral)
        batch_tgt_zh = translate_batch(batch_tgt_en, tokenizer, model, device)

        zh_src.extend(batch_src_zh)
        zh_tgt.extend(batch_tgt_zh)

    # Save new DataFrame
    df_zh = pd.DataFrame(
        {
            "src_raw": zh_src,
            "tgt_raw": zh_tgt,
            "src_en_original": df["src_raw"],  # Keep original for reference
            "tgt_en_original": df["tgt_raw"],
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_zh.to_csv(output_path, index=False)
    print(f"Saved {len(df_zh)} translated rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed/")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer, model = load_translator(device)

    files_to_translate = [
        ("train_complex.csv", "train_chinese_synthetic.csv"),
        ("val_complex.csv", "val_chinese_synthetic.csv"),
        # We generally DO NOT translate the test set for training.
        # We want to test on 'Gold' Chinese data, not synthetic data.
        # But for 'Synthetic Validation', we can translate val.
    ]

    for input_name, output_name in files_to_translate:
        in_path = os.path.join(args.data_dir, input_name)
        out_path = os.path.join(args.data_dir, output_name)
        process_file(in_path, out_path, tokenizer, model, device)
