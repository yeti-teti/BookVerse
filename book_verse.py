# Import necessary libraries
import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.to(device)

# Load the CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Preprocessing the data
def preprocess_function(examples):
    inputs = examples['article']
    targets = examples['highlights']

    model_inputs = tokenizer(
        inputs,
        max_length=1024,
        padding='max_length',
        truncation=True,
    )

    labels = tokenizer(
        targets,
        max_length=256,
        padding='max_length',
        truncation=True,
    )

    # Replace padding token id's in labels by -100 so they are ignored in the loss computation
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
        for labels_example in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the preprocessing
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['train'].column_names,
)

# Randomly shuffle and select 50,000 examples
total_examples = 10000
# Preprocessing all splits
tokenized_datasets = {
    split: dataset[split].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset[split].column_names,
    )
    for split in dataset.keys()
}

# Shuffle and select on the 'train' split
train_size = int(0.8 * total_examples)
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(train_size))
eval_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(train_size, total_examples))


# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding='max_length',
    max_length=1024,
)

# Set training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./bart-fine-tuned-model",
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    per_device_train_batch_size=2,  # Adjust based on your GPU memory
    per_device_eval_batch_size=2,
    num_train_epochs=1,  # Increase this for better results
    fp16=True if torch.cuda.is_available() else False,  # Enable if using GPU with FP16 support
    logging_steps=250,
    learning_rate=3e-5,
    predict_with_generate=True,
    save_total_limit=2,
    dataloader_num_workers=4,
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the model
trainer.save_model("./bart-fine-tuned-model")
tokenizer.save_pretrained("./bart-fine-tuned-model")
