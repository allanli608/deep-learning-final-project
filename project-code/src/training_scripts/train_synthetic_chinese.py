import pandas as pd  # Make sure to import pandas at the top
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from src import MBartNeutralizer, WNCDataset, WeightedSeq2SeqTrainer

print("--- INITIALIZING PHASE 3: SYNTHETIC CHINESE TRAINING ---")

# 1. Initialize mBART
neutralizer = MBartNeutralizer(
    model_name="facebook/mbart-large-50", 
    src_lang="zh_CN", 
    tgt_lang="zh_CN"
)
model = neutralizer.get_model()
tokenizer = neutralizer.get_tokenizer()

# 2. Load Data
train_path = "data/processed/train_chinese_synthetic.csv"
val_path = "data/processed/val_chinese_synthetic.csv"

print(f"Loading datasets from {train_path}...")
train_set = WNCDataset(train_path, tokenizer)
val_set = WNCDataset(val_path, tokenizer)

# 3. Configure Training
training_args = Seq2SeqTrainingArguments(
    output_dir="./results_zh_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=100,
    save_steps=1500,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=1500,
    fp16=True,
    remove_unused_columns=False,
)

# 4. Initialize Trainer
trainer = WeightedSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

# 5. Train
print("Starting training for Model 3 (Synthetic Chinese)...")
trainer.train()

# 6. Save Final Model
output_path = "models/mbart_neutralizer_zh_synthetic"
neutralizer.save_model(output_path)
print(f"Phase 3 Complete. Model saved to {output_path}")

# 7. EXPORT LOGS (The Safety Net)
print("Saving training logs to CSV...")
log_history = trainer.state.log_history
df_log = pd.DataFrame(log_history)
log_file_path = os.path.join(training_args.output_dir, "training_log.csv")
df_log.to_csv(log_file_path, index=False)
print(f"Full logs saved to: {log_file_path}")