import os
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset

# path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# load dataset from the same directory
try:
    train_df = pd.read_csv("datasets/t5_train_dataset.csv")
    test_df = pd.read_csv("datasets/t5_test_dataset.csv")
except FileNotFoundError as e:
    print("Error: Error in finding training/testing dataset for CodeT5.")
    print(f"Error details: {e}")
    exit()

# Debug: Print sample
print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")
print(f"Sample Input: {train_df.iloc[0]['input_text']}")
print(f"Sample Target: {train_df.iloc[0]['target_text']}")
print(f"Sample Intention: {train_df.iloc[0]['intention']}")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

print("Loading model and tokenizer...")
model_name = "Salesforce/codet5-base"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model {model_name}: {e}")
    sys.exit(1)

max_input_length = 128
max_target_length = 128

def preprocess_function(examples):
    """Every row in the dataset has an input code snippet and a target fixed code snippet. and every row will go through this function"""
    # Add prefix "fix: " to assist the model (common T5 practice)
    # Check if intention is available, if so, include it as context
    inputs = []
    has_intention = "intention" in examples
    for i in range(len(examples["input_text"])):
        code = examples["input_text"][i]

        if has_intention:
            intention = examples["intention"][i]
        else:
            intention = ""
        
        # Check if intention is valid (not None or some bad value)
        if pd.isna(intention) or not str(intention).strip():
            inputs.append("fix: " + str(code))
        else:
            inputs.append(f"fix intent: {str(intention).strip()} code: {str(code)}")
            
    targets = [str(code) for code in examples["target_text"]]
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, padding="max_length", truncation=True)

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
    use_cpu=True, # Force CPU because the local GPU 1070 is not supported by the installed PyTorch build
    fp16=False,
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
