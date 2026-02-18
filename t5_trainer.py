import os
import sys
import pandas as pd
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
# Parent dir (training/)
base_dir = os.path.dirname(script_dir)
train_csv = os.path.join(base_dir, "t5_train_dataset.csv")
test_csv = os.path.join(base_dir, "t5_test_dataset.csv")

if not os.path.exists(train_csv):
    print(f"Error: {train_csv} not found.")
    sys.exit(1)

print("Loading data...")
try:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
except Exception as e:
    print(f"Error loading CSVs: {e}")
    sys.exit(1)
print("data loaded")

# Debug: Print sample
print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")
print(f"Sample Input: {train_df.iloc[0]['input_text']}")
print(f"Sample Target: {train_df.iloc[0]['target_text']}")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

print("Loading model and tokenizer...")
model_name = "Salesforce/codet5-base"
try:
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model {model_name}: {e}")
    print("Please ensure you have internet access to download the model.")
    sys.exit(1)

max_input_length = 128
max_target_length = 128

def preprocess_function(examples):
    # Add prefix "fix: " to assist the model (common T5 practice)
    inputs = ["fix: " + str(code) for code in examples["input_text"]]
    targets = [str(code) for code in examples["target_text"]]
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)
    
    # Process targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding="max_length", truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# Define arguments
output_dir = os.path.join(script_dir, "results")
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    eval_steps=1000, # Evaluate less frequently to save time
    learning_rate=2e-5,
    per_device_train_batch_size=4, # Conservative batch size
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=1, # 1 Epoch is usually enough for a demo on this data size
    predict_with_generate=True,
    logging_dir=os.path.join(script_dir, "logs"),
    logging_steps=100,
    fp16=torch.cuda.is_available(), # Use mixed precision if CUDA available
    push_to_hub=False,
    dataloader_num_workers=0, # Windows safety: avoid multiprocessing spawn issues
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()

print("Saving model...")
save_path = os.path.join(script_dir, "saved_model")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved to {save_path}")