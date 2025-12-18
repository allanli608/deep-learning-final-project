import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from src import MBartNeutralizer, WNCDataset, WeightedSeq2SeqTrainer

print(f"GPU Available: {torch.cuda.is_available()}")

neutralizer = MBartNeutralizer(model_name="facebook/mbart-large-50")
model = neutralizer.get_model()
tokenizer = neutralizer.get_tokenizer()

# Load the filtered "Complex" dataset created by preprocess.py
train_set = WNCDataset("data/processed/train_complex.csv", tokenizer)
val_set = WNCDataset("data/processed/val_complex.csv", tokenizer)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,  # Adjust based on your GPU VRAM
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,  # Lower LR for fine-tuning
    logging_steps=100,
    save_steps=1500,
    eval_strategy="steps",
    eval_steps=1500,
    fp16=True,  # Essential for mBART memory efficiency
    remove_unused_columns=False,  # IMPORTANT: Keep 'loss_weights' in the batch
)

trainer = WeightedSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

train_result = trainer.train()

# Save the fine-tuned weights
neutralizer.save_model("models/mbart_neutralizer_en_v1")

# Quick Inference Test
input_text = "The radical regime failed to act."
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

print("Loading fine-tuned model for final verification...")
# Make sure this matches the path you saved to
saved_path = "models/mbart_neutralizer_en_v1"
neutralizer = MBartNeutralizer(model_name=saved_path)
model = neutralizer.get_model()
tokenizer = neutralizer.get_tokenizer()

# 2. Define "The Gauntlet" (Test Cases)
test_cases = [
    # Case 1: Subjective Intensifier (Easy)
    "The radical regime failed to act on the crisis.",
    # Case 2: Framing Bias (Harder - subtle verb change)
    "The controversial politician foolishly denied the allegations.",
    # Case 3: Presupposition (Hardest - implies guilt)
    "He exposed the senator's corruption.",
]

# 3. Run Robust Inference
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"\n{'='*20} PHASE 1 COMPLETE: ENGLISH BASELINE {'='*20}\n")

for text in test_cases:
    # A. Tokenize (Force English Source)
    tokenizer.src_lang = "en_XX"
    encoded = tokenizer(text, return_tensors="pt").to(device)

    # B. Generate (Prevent Repetition & Force English Output)
    generated_ids = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
        max_length=64,
        num_beams=5,  # Smarter search
        no_repeat_ngram_size=2,  # Prevents "same same" loops
        repetition_penalty=1.2,  # Soft penalty to encourage natural phrasing
        early_stopping=True,
    )

    # C. Decode
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # D. Display
    print(f"Original: {text}")
    print(f"Neutral:  {output}")
    print("-" * 50)

print(
    "\nIf the 'Neutral' outputs removed the biased words (radical, foolishly, exposed)"
)
print("while keeping the facts, Phase 1 is SUCCESSFUL.")
