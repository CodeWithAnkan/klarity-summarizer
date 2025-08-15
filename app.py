from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer

app = Flask(__name__)

# Load the model and its specific tokenizer for precise truncation
model_name = "sshleifer/distilbart-cnn-12-6"
print("Loading summarization model and tokenizer...")
summarizer = pipeline("summarization", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Summarization model and tokenizer loaded successfully.")

# --- THIS IS THE FIX ---
# Add a simple health check endpoint at the root URL
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200
# --- END OF FIX ---

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body'}), 400

    text_to_summarize = data['text']
    
    # Use the tokenizer to safely truncate the text to the model's exact max length
    inputs = tokenizer(text_to_summarize, max_length=1024, return_tensors="pt", truncation=True)
    safe_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    try:
        summary = summarizer(safe_text, max_length=150, min_length=40, do_sample=False)
        summary_text = summary[0]['summary_text']
        return jsonify({'summary_text': summary_text})
        
    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({'error': 'Failed to summarize text'}), 500

if __name__ == '__main__':
    app.run(port=5002)