import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import os

# CONFIG
INPUT_FILE = "data/processed/test_complex.csv"
OUTPUT_FILE = "data/processed/test_chinese_gold.csv"
SAMPLE_SIZE = 100  # Number of examples to grab
MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"


def load_translator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Translator on {device}...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    # Use FP16 for speed
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(device).half()
    model.eval()
    return tokenizer, model, device


def translate_list(text_list, tokenizer, model, device, batch_size=32):
    translated_texts = []
    print(f"Translating {len(text_list)} lines...")

    for i in tqdm(range(0, len(text_list), batch_size)):
        batch = text_list[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            # Greedy search is fine for pseudo-gold
            generated = model.generate(**inputs, num_beams=1, max_new_tokens=128)

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translated_texts.extend(decoded)

    return translated_texts


if __name__ == "__main__":
    # 1. Load English Data
    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 2. Sample 100 Random pairs
    # We use a fixed random_state so this set is consistent every time you run it
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)

    # 3. Translate
    tokenizer, model, device = load_translator()

    print("Translating Biased (Source)...")
    zh_biased = translate_list(df_sample["src_raw"].tolist(), tokenizer, model, device)

    print("Translating Neutral (Target)...")
    zh_neutral = translate_list(df_sample["tgt_raw"].tolist(), tokenizer, model, device)

    # 4. Format for evaluate.py
    # Expected columns: | ID | English_Ref | Chinese_Biased | Chinese_Neutral_Gold |
    gold_df = pd.DataFrame(
        {
            "ID": range(len(df_sample)),
            "English_Ref": df_sample["src_raw"].tolist(),
            "Chinese_Biased": zh_biased,
            "Chinese_Neutral_Gold": zh_neutral,
        }
    )

    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    gold_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUCCESS: Pseudo-Gold set saved to {OUTPUT_FILE}")
    print("Columns:", gold_df.columns.tolist())
    print("Sample Row:\n", gold_df.iloc[0])
