import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pandas as pd  # Make sure to import pandas at the top
import math

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from src import MBartNeutralizer, WNCDataset, WeightedSeq2SeqTrainer

print("--- INITIALIZING PHASE 4: LOW-RESOURCE (FEW-SHOT) EXPERIMENT ---")

# ==========================================
# CONFIGURATION (CHANGE THIS NUMBER)
# ==========================================
DATA_FRACTION = 0.01  # 0.01 = 1% (Few-Shot), 0.10 = 10% (Low-Resource)
# ==========================================

# 1. Prepare the Data Subset
full_train_path = "data/processed/train_chinese_synthetic.csv"
subset_train_path = f"data/processed/train_chinese_{int(DATA_FRACTION*100)}percent.csv"

print(f"Creating {DATA_FRACTION*100}% Data Sample...")
df_full = pd.read_csv(full_train_path)

# Sampling with fixed random_state for reproducibility (Crucial for papers!)
df_subset = df_full.sample(frac=DATA_FRACTION, random_state=42)
df_subset.to_csv(subset_train_path, index=False)
print(f"Saved {len(df_subset)} rows to {subset_train_path}")

# 2. Dynamic Training Config
# Rule of thumb: If we reduce data by 100x (1%), we should increase epochs
# to ensure the model sees enough updates.
# We'll calculate epochs to target roughly ~1000-2000 total steps.
batch_size = 4
grad_acc = 8
effective_batch = batch_size * grad_acc
steps_per_epoch = len(df_subset) // effective_batch

# Ensure we train for at least a reasonable number of steps (e.g., 500)
# If 1% data (150 rows) -> 4 steps per epoch.
# We need ~100 epochs to get 400 steps.
target_total_steps = 600
calculated_epochs = max(10, math.ceil(target_total_steps / max(1, steps_per_epoch)))

print(
    f"Calculated training for {calculated_epochs} epochs (approx {calculated_epochs * steps_per_epoch} steps)"
)

# 3. Initialize mBART
neutralizer = MBartNeutralizer(
    # CHANGE THIS: Point to your English model folder
    model_name="models/mbart_neutralizer_en_v1", 
    src_lang="zh_CN", 
    tgt_lang="zh_CN"
)
model = neutralizer.get_model()
tokenizer = neutralizer.get_tokenizer()

# 4. Load Data
train_set = WNCDataset(subset_train_path, tokenizer)
# Note: We keep the FULL validation set to make metrics comparable to the Full Model
val_set = WNCDataset("data/processed/val_chinese_synthetic.csv", tokenizer)

# 5. Configure Training
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./results_zh_{int(DATA_FRACTION*100)}percent",  # Distinct folder
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_acc,
    gradient_checkpointing=True,
    num_train_epochs=calculated_epochs,  # Use our scaled epoch count
    learning_rate=2e-5,
    # Logging/Saving must be frequent for small data
    logging_steps=10,
    save_steps=1000,
    save_total_limit=1,
    eval_strategy="steps",
    eval_steps=1000,
    fp16=True,
    remove_unused_columns=False,
)

# 6. Initialize Trainer
trainer = WeightedSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

# 7. Train
print(f"Starting training for Model 4 ({DATA_FRACTION*100}% Data)...")
trainer.train()

# 8. Save Final Model
output_path = f"models/mbart_neutralizer_zh_{int(DATA_FRACTION*100)}percent"
neutralizer.save_model(output_path)
print(f"Phase 4 Complete. Model saved to {output_path}")

# 9. EXPORT LOGS
print("Saving training logs to CSV...")
log_history = trainer.state.log_history
df_log = pd.DataFrame(log_history)
log_file_path = os.path.join(training_args.output_dir, "training_log.csv")
df_log.to_csv(log_file_path, index=False)
print(f"Full logs saved to: {log_file_path}")
