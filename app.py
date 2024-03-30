# flask app for text summarization purposes
from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        input_text = request.form['input_text']

        summarizer = pipeline (
            task="summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn",
            framework="pt",
            min_length=20,
            max_length=40,
            truncation=True,
            model_kwargs={"cache_dir": '/Documents/Huggin_Face/'},
        )
        
        output = summarizer(input_text, max_length=150, min_length=30, do_sample=False)
        summary = output[0]['summary_text']

        return render_template('summary.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)