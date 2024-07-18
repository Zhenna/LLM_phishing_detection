from flask import Flask, request, render_template 
from src.infer import make_inference

app = Flask(__name__)

@app.route('/')
def index():
    """show a web page for user input"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """make an inference using either LLM or xgboost"""

    # Retrieve the text from index page
    text = request.form.get('message') 

    # make inference
    pred = make_inference(text).best_llm()
    # pred = make_inference(text).best_baseline()

    outcome = ""
    if pred == 0:
        outcome = " not"

    return render_template('predict.html', message=text, pred=pred, is_phising=outcome)

if __name__ == '__main__':
    # app.run(debug=True) 
    app.run(host="0.0.0.0", port=5000)