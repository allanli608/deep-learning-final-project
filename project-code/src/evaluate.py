import torch
import pandas as pd
import numpy as np
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    GPT2LMHeadModel,
    BertTokenizerFast,
)
from bert_score import score
import math


class Evaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Evaluators on {self.device}...")

        # 1. Load Style Judge
        try:
            self.judge_tokenizer = BertTokenizer.from_pretrained("models/bert_judge_zh")
            self.judge_model = BertForSequenceClassification.from_pretrained(
                "models/bert_judge_zh"
            ).to(self.device)
            self.judge_model.eval()
        except OSError:
            print(
                "⚠️ WARNING: Style Judge not found at 'models/bert_judge_zh'. Style Accuracy will be 0."
            )
            self.judge_model = None

        # 2. Load Fluency Judge (GPT2-Chinese)
        self.gpt_tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
        self.gpt_model = GPT2LMHeadModel.from_pretrained(
            "ckiplab/gpt2-base-chinese"
        ).to(self.device)
        self.gpt_model.eval()

    def get_style_accuracy(self, texts):
        """Returns % of texts classified as Neutral (Label 0)"""
        if self.judge_model is None:
            return 0.0

        inputs = self.judge_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            logits = self.judge_model(**inputs).logits
            preds = torch.argmax(logits, dim=1)

        neutral_count = (preds == 0).sum().item()
        return neutral_count / len(texts)

    def get_perplexity(self, texts):
        """Calculates PPL (Fluency). Lower is better."""
        ppls = []
        # Process one by one to avoid OOM on GPT2
        for text in texts:
            if not text.strip():
                continue  # Skip empty
            encodings = self.gpt_tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.gpt_model(**encodings, labels=encodings.input_ids)
                loss = outputs.loss
                ppl = math.exp(loss.item())
                if not math.isnan(ppl):
                    ppls.append(ppl)
        return np.mean(ppls) if ppls else float("inf")

    def get_bert_score(self, candidates, references):
        """Calculates Content Preservation (F1). Higher is better."""
        # BERTScore handles batching internally, but for safety we can just pass the lists
        try:
            P, R, F1 = score(
                candidates,
                references,
                lang="zh",
                verbose=False,
                device=self.device,
                batch_size=32,
            )
            return F1.mean().item()
        except Exception as e:
            print(f"BERTScore Error: {e}")
            return 0.0

    def calculate_composite_score(self, acc, bert_score, ppl):
        """
        Geometric Mean of (Accuracy * Preservation * 1/PPL)
        Since PPL is usually > 1, we use 100/PPL to scale it to a similar magnitude as Acc/F1
        for better readability, or just 1/PPL.
        Standard Formula: (Acc * BERTScore * (1/log(PPL)))^(1/3) is common,
        but let's use the simple GM: (Acc * BERTScore * (1/PPL))^(1/3)
        """
        # Avoid division by zero
        if ppl <= 0:
            ppl = 9999

        fluency_component = 1 / ppl

        # We take the cube root of the product
        product = acc * bert_score * fluency_component
        gm = product ** (1 / 3)
        return gm
