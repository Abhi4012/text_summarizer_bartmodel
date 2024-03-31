from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os

app = Flask(__name__)

# Cache directory ko badalne ka tareeka
cache_dir = os.path.join(app.instance_path, 'cache')

# Cache directory ko banane ka tareeka
os.makedirs(cache_dir, exist_ok=True)

# Tokenizer aur model ko alag alag load karne ka tareeka
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn", cache_dir=cache_dir)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        input_text = request.form['input_text']

        summarizer = pipeline(
            task="summarization",
            model=model,
            tokenizer=tokenizer,
            framework="pt",
            min_length=20,
            max_length=40,
            truncation=True,
        )
        
        output = summarizer(input_text, max_length=150, min_length=30, do_sample=False)
        summary = output[0]['summary_text']

        return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
