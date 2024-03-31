# from flask import Flask, render_template, request, jsonify
# from transformers import pipeline

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/summarize', methods=['POST'])
# def summarize():
#     if request.method == 'POST':
#         input_text = request.form['input_text']

#         summarizer = pipeline (
#             task="summarization",
#             model="facebook/bart-large-cnn",
#             tokenizer="facebook/bart-large-cnn",
#             framework="pt",
#             min_length=20,
#             max_length=40,
#             truncation=True,
#             model_kwargs={"cache_dir": '/Documents/Huggin_Face/'},
#         )
        
#         output = summarizer(input_text, max_length=150, min_length=30, do_sample=False)
#         summary = output[0]['summary_text']

#         return jsonify({'summary': summary})


# # Serving the static files
# @app.route('/static/<path:filename>')
# def serve_static(filename):
#     return send_from_directory('static', filename)

# # Serving the templates
# @app.route('/<template_name>')
# def serve_template(template_name):
#     return render_template(f'{template_name}.html')


from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Load tokenizer and model separately to avoid caching issues
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

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
