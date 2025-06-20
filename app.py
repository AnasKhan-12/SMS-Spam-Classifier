from flask import Flask, request, render_template
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import os

nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

nltk.download('punkt')       # Tokenizer
nltk.download('stopwords')   # Stopword list


stemmer = PorterStemmer()

def transform(text):
    text = text.lower() 
    text = nltk_download_data.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():  # Only letters and numbers
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:  # Remove stopwords and punctuation
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    
    for i in text:
        y.append(stemmer.stem(i))

    return " ".join(y)
def predict_sms(msg):
    cleaned = transform(msg)
    vect_msg = vectorizer.transform([cleaned])
    result = model.predict(vect_msg)[0]
    return 'Spam' if result == 1 else 'Not Spam'


app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        result = predict_sms(message)
        print("USER INPUT:", message)
        print("PREDICTION:", result)
        return render_template('index.html', prediction_text=result)




if __name__ == '__main__':
    app.run(debug=True)





