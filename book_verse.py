# Importing Packages
import os
import re
import torch
import pdfplumber
import bitsandbytes as bnb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset

# ------------------------------------------------ #
# Extract Text from PDF

def extract_text_from_pdf(pdf_path, txt_output_path):
    print(f"Extracting text from {pdf_path}...")
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                # Clean up extracted text
                page_text = re.sub(r'\s+', ' ', page_text)  # Remove extra whitespaces
                page_text = re.sub(r'(\w)-\n(\w)', r'\1\2', page_text)  # Remove hyphenation
                text += page_text + " "
    # Additional text cleaning
    text = re.sub(r'\n+', ' ', text)
    text = text.strip()
    # Save extracted text to TXT file
    with open(txt_output_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)
    print(f"Text extracted and saved to {txt_output_path}")

# ------------------------------------------------ #
# Prompts

def generate_prompt(text):
    return f"Summarize the following text:\n\n{text}\n\nSummary:"

# ------------------------------------------------ #
# Loading the model and Quantizing

# Base model
base_model_name = "meta-llama/Meta-Llama-3.1-8B-instruct"  # Ensure this model is available

# Quantizing the model for efficient fine-tuning
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)

model.config.use_cache = False

# ------------------------------------------------ #
# Tokenization

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

pipe = pipeline(
    task="summarization",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1
)

# ------------------------------------------------ #
# Extracting the linear modules names and setting up the model

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)
print(f"Linear modules for the model {modules}")

# Setting up the model
output_dir = "llama-fine-tuned-model"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules,
)

training_arguments = TrainingArguments(
    output_dir=output_dir,                    # Directory to save checkpoints
    num_train_epochs=1,                       # Number of training epochs
    per_device_train_batch_size=1,            # Batch size per device during training
    gradient_accumulation_steps=8,            # Number of steps before performing a backward/update pass
    gradient_checkpointing=True,              # Use gradient checkpointing to save memory
    optim="paged_adamw_32bit",
    logging_steps=10,
    learning_rate=2e-4,                       # Learning rate, based on QLoRA paper
    weight_decay=0.001,
    fp16=True,
    max_grad_norm=0.3,                        # Max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                        # Warmup ratio based on QLoRA paper
    group_by_length=False,
    lr_scheduler_type="cosine",               # Use cosine learning rate scheduler
    evaluation_strategy="steps",              # Evaluate every few steps
    eval_steps=500,                           # Evaluation frequency
    save_steps=500,                           # Save checkpoint frequency
    logging_dir='./logs',                     # Directory for storing logs
)

# ------------------------------------------------ #
# Loading and Preprocessing Dataset

# Use a summarization dataset, e.g., CNN/DailyMail
dataset = load_dataset('cnn_dailymail', '3.0.0')

def preprocess_function(examples):
    inputs = examples['article']
    targets = examples['highlights']
    texts = [generate_prompt(input_text) + ' ' + target for input_text, target in zip(inputs, targets)]
    return {'text': texts}

train_data = dataset['train'].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['train'].column_names
)
eval_data = dataset['validation'].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['validation'].column_names
)

# ------------------------------------------------ #
# Training the model

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=512,
    packing=False,
    dataset_kwargs={
        "add_special_tokens": True,
        "append_eos_token": True,
    }
)

trainer.train()

# Saving the trained Model and Tokenizer
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# ------------------------------------------------ #
# Summarization Function using the Saved Model

def summarize_text(text):
    # Load the fine-tuned model and tokenizer
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    fine_tuned_tokenizer.pad_token_id = fine_tuned_tokenizer.eos_token_id

    # Prepare the input prompt
    input_prompt = generate_prompt(text)
    inputs = fine_tuned_tokenizer(input_prompt, return_tensors="pt").to('cpu')

    # Generate the summary
    output = fine_tuned_model.generate(
        input_ids=inputs['input_ids'],
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.2,
        eos_token_id=fine_tuned_tokenizer.eos_token_id,
        pad_token_id=fine_tuned_tokenizer.eos_token_id,
    )
    summary = fine_tuned_tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract the summary after "Summary:"
    if "Summary:" in summary:
        summary = summary.split("Summary:")[1].strip()
    return summary

# ------------------------------------------------ #
# Extracting Text from PDF and Summarizing

# Specify the paths
pdf_path = 'Paper-draft.pdf'            # Replace with your PDF file path
txt_output_path = 'extracted_text.txt'    # New file where PDF text will be saved

# Extract text from the PDF and save to a new file
extract_text_from_pdf(pdf_path, txt_output_path)

# Read the extracted text
with open(txt_output_path, 'r', encoding='utf-8') as file:
    pdf_text = file.read()

# Use the fine-tuned model to summarize the extracted text
summary = summarize_text(pdf_text)

# Output the summary
print("Summary of the PDF content:")
print(summary)

# Optionally, save the summary to a new text file
summary_output_path = 'pdf_summary.txt'
with open(summary_output_path, 'w', encoding='utf-8') as file:
    file.write(summary)
print(f"Summary saved to {summary_output_path}")
