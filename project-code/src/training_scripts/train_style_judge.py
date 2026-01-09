import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def prepare_data(filepath):
    """
    Turns your paired translation data into a classification dataset.
    Source (Biased) -> Label 1
    Target (Neutral) -> Label 0
    """
    print(f"Loading data for Judge training: {filepath}")
    df = pd.read_csv(filepath)

    # 1. Biased Samples (Label = 1)
    biased_texts = df["src_raw"].astype(str).tolist()
    biased_labels = [1] * len(biased_texts)

    # 2. Neutral Samples (Label = 0)
    neutral_texts = df["tgt_raw"].astype(str).tolist()
    neutral_labels = [0] * len(neutral_texts)

    # Combine
    texts = biased_texts + neutral_texts
    labels = biased_labels + neutral_labels

    return train_test_split(texts, labels, test_size=0.1, random_state=42)


if __name__ == "__main__":
    MODEL_NAME = "google-bert/bert-base-chinese"

    # 1. Prepare Data
    # We use the synthetic data because it's the only Chinese pairs we have!
    train_texts, val_texts, train_labels, val_labels = prepare_data(
        "data/processed/train_chinese_synthetic.csv"
    )

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer)
    val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer)

    # 2. Setup Model
    print("Initializing BERT Judge...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # 3. Train
    training_args = TrainingArguments(
        output_dir="./results/results_judge",
        num_train_epochs=2,  # Short training is fine for a classifier
        per_device_train_batch_size=32,  # BERT-base is small, 32 fits easily
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=2e-5,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    # 4. Save
    model.save_pretrained("models/bert_judge_zh")
    tokenizer.save_pretrained("models/bert_judge_zh")
    print("Judge Saved to models/bert_judge_zh")
