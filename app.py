# flask app for text summarization purposes
from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html') # this is index.html render 

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        input_text = request.form['input_text']

        summarizer = pipeline (
            task="summarization",
            model="facebook/bart-large-cnn", # using this model form huggingface https://huggingface.co/facebook/bart-large-cnn
            tokenizer="facebook/bart-large-cnn", # word tokenizer using facebook/bart-large-cnn model
            framework="pt",
            min_length=20,
            max_length=40,
            truncation=True,
            model_kwargs={"cache_dir": '/Documents/Huggin_Face/'},
        )
        
        output = summarizer(input_text, max_length=150, min_length=30, do_sample=False)
        summary = output[0]['summary_text']

        return render_template('summary.html', summary=summary) # This is template/summary.html which get summary from text that you will provide to app

if __name__ == '__main__':
    app.run(debug=True)