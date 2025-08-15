from transformers import pipeline, AutoTokenizer

# This script will download the model and tokenizer to a local directory
model_name = "sshleifer/distilbart-cnn-12-6"
output_dir = "./model_files"

print(f"Downloading model '{model_name}' to '{output_dir}'...")

# Download and save the model and tokenizer
summarizer = pipeline("summarization", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

summarizer.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Model download complete.")