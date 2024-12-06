# Import necessary libraries
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import pdfplumber
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and tokenizer
model_dir = "./bart-fine-tuned-model"  # Update to the directory where your trained model is saved

tokenizer = BartTokenizer.from_pretrained(model_dir)
model = BartForConditionalGeneration.from_pretrained(model_dir)
model.to(device)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from {pdf_path}...")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                # Clean up extracted text
                page_text = re.sub(r'\s+', ' ', page_text)  # Remove extra whitespaces
                page_text = page_text.strip()
                text += page_text + " "  # Add a space to separate pages
            else:
                print(f"Warning: No text found on page {page_number}.")
    text = text.strip()
    return text

# Function to chunk text into manageable sizes
def chunk_text(text, max_tokens=1024):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ''
    current_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer.tokenize(sentence))
        # If adding the sentence exceeds the max token limit, save the current chunk and start a new one
        if current_length + sentence_length > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = sentence_length
        else:
            current_chunk += ' ' + sentence
            current_length += sentence_length

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Function to summarize a chunk
def summarize_chunk(chunk_text, min_length=400, max_length=1000):
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        chunk_text,
        return_tensors='pt',
        max_length=1024,
        truncation=True,
    )

    # Move tensors to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Generate summary
    summary_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=4,
        length_penalty=2.0,
        min_length=min_length,
        max_length=max_length,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Main function to summarize PDF
def summarize_pdf(pdf_path, output_summary_path):
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("No text extracted from the PDF.")
        return

    # Chunk the text into manageable sizes
    chunks = chunk_text(text)
    print(f"Total chunks created: {len(chunks)}")

    # Summarize each chunk
    chunk_summaries = []
    for idx, chunk in enumerate(chunks):
        print(f"Summarizing chunk {idx+1}/{len(chunks)}...")
        try:
            # Use 400-1000 word summary range
            summary = summarize_chunk(chunk, min_length=400, max_length=1000)
            chunk_summaries.append(summary)
        except Exception as e:
            print(f"An error occurred while summarizing chunk {idx+1}: {e}")

    if not chunk_summaries:
        print("No summaries were generated.")
        return

    # Combine summaries
    combined_summary = ' '.join(chunk_summaries)

    # Refine summary to target word count
    current_summary = combined_summary
    iteration = 1
    while True:
        word_count = len(current_summary.split())
        print(f"Iteration {iteration}: Current summary length is {word_count} words.")

        # Check if summary is within the desired range
        if 400 <= word_count <= 1000:
            print("Desired summary length achieved.")
            break

        # If summary is too long, compress it
        if word_count > 1000:
            print("Summary too long. Compressing...")
            try:
                current_summary = summarize_chunk(current_summary, min_length=400, max_length=1000)
            except Exception as e:
                print(f"An error occurred during iteration {iteration}: {e}")
                break
        
        # If summary is too short, try to expand it slightly
        elif word_count < 400:
            print("Summary too short. Attempting to expand...")
            try:
                current_summary = summarize_chunk(current_summary, min_length=400, max_length=1000)
            except Exception as e:
                print(f"An error occurred during iteration {iteration}: {e}")
                break

        iteration += 1

        # Break if max iterations reached to prevent infinite loop
        if iteration > 5:
            print("Maximum iterations reached.")
            break

    # Save the summary
    with open(output_summary_path, 'w', encoding='utf-8') as f:
        f.write(current_summary)

    print(f"Summary saved to {output_summary_path}")

# Example usage
if __name__ == "__main__":
    pdf_path = 'WNF.pdf'  # Replace with the path to your PDF
    output_summary_path = 'book_summary.txt'

    summarize_pdf(pdf_path, output_summary_path)