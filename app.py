from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer

app = Flask(__name__)

# --- THE FIX: Load the model AND its tokenizer ---
model_name = "sshleifer/distilbart-cnn-12-6"
print("Loading summarization model and tokenizer...")
summarizer = pipeline("summarization", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Summarization model and tokenizer loaded successfully.")

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body'}), 400

    text_to_summarize = data['text']
    
    # --- THIS IS THE ROBUST FIX ---
    # Use the tokenizer to truncate the text to the model's exact max length
    # We use 1024 as the limit for this BART-family model
    inputs = tokenizer(text_to_summarize, max_length=1024, return_tensors="pt", truncation=True)
    text_to_summarize = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    # --- END OF FIX ---

    try:
        summary = summarizer(text_to_summarize, max_length=150, min_length=40, do_sample=False)
        summary_text = summary[0]['summary_text']
        return jsonify({'summary_text': summary_text})
        
    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({'error': 'Failed to summarize text'}), 500

if __name__ == '__main__':
    app.run(port=5002)