from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# --- THE FIX: Use a smaller, more reliable model ---
print("Loading T5-small summarization model...")
summarizer = pipeline("summarization", model="t5-small")
print("Summarization model loaded successfully.")

# Add a simple health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body'}), 400

    text_to_summarize = data['text']

    try:
        summary = summarizer(text_to_summarize, max_length=150, min_length=40, do_sample=False)
        summary_text = summary[0]['summary_text']
        return jsonify({'summary_text': summary_text})
        
    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({'error': 'Failed to summarize text'}), 500

if __name__ == '__main__':
    app.run(port=5002)