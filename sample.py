# sample.py

# Importing Packages
import re
import torch
import pdfplumber
import bitsandbytes as bnb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig

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
# Summarization Function using the Saved Model

def summarize_text(text):
    # Load the fine-tuned model and tokenizer
    output_dir = "llama-fine-tuned-model"

    # Load config
    peft_config = PeftConfig.from_pretrained(output_dir)

    # Base model
    base_model_name = peft_config.base_model_name_or_path

    # Quantizing the model for efficient inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )

    # Load the PEFT model
    model = PeftModel.from_pretrained(base_model, output_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare the input prompt
    input_prompt = generate_prompt(text)
    inputs = tokenizer(input_prompt, return_tensors="pt").to('cpu')

    # Generate the summary
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract the summary after "Summary:"
    if "Summary:" in summary:
        summary = summary.split("Summary:")[1].strip()
    return summary

# ------------------------------------------------ #
# Extracting Text from PDF and Summarizing

if __name__ == "__main__":
    # Specify the paths
    pdf_path = 'Paper-draft.pdf'            # Replace with your PDF file path
    txt_output_path = 'extracted_text.txt'  # New file where PDF text will be saved

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
