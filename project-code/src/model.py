# src/model.py
import torch

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


class MBartNeutralizer:
    def __init__(
        self, model_name="facebook/mbart-large-50", src_lang="en_XX", tgt_lang="en_XX"
    ):
        """
        Wrapper for mBART-50 to handle Subjective Bias Neutralization.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing mBART on {self.device}...")

        # Load Tokenizer
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            model_name, src_lang=src_lang, tgt_lang=tgt_lang
        )

        # Load Model
        self.model = MBartForConditionalGeneration.from_pretrained(model_name, attn_implementation="eager")
        self.model.to(self.device)

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
