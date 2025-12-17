# src/train.py
import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
from transformers import Seq2SeqTrainer


# baseline loss function
class WNCDataset(Dataset):
    """
    Custom Dataset that tokenizes text and calculates 'loss_weights'.
    """

    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src_text = str(row["src_raw"])
        tgt_text = str(row["tgt_raw"])

        # Tokenize Inputs
        inputs = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize Targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        
        # Get the label IDs
        labels_ids = labels["input_ids"].squeeze() 
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels_ids, # Return the masked labels
        }


class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom Trainer that implements Weighted Cross Entropy Loss.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Extract weights and remove them from inputs so model doesn't crash
        loss_weights = inputs.pop("loss_weights", None)

        # Forward pass
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if loss_weights is not None:
            # Flatten tensors for CrossEntropyLoss
            logits_flat = logits.view(-1, self.model.config.vocab_size)
            labels_flat = labels.view(-1)
            weights_flat = loss_weights.view(-1)

            # Compute weighted loss
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            raw_loss = loss_fct(logits_flat, labels_flat)

            # Apply weights and normalize
            weighted_loss = raw_loss * weights_flat
            final_loss = weighted_loss.sum() / (weights_flat.sum() + 1e-9)
        else:
            loss_fct = nn.CrossEntropyLoss()
            final_loss = loss_fct(
                logits.view(-1, self.model.config.vocab_size), labels.view(-1)
            )

        return (final_loss, outputs) if return_outputs else final_loss
