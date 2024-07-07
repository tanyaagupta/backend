from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import nltk
import pickle
import pandas as pd
from spacy.tokens import Token

Token.set_extension('lemma', default=None)
nlp = spacy.load('en_core_web_sm')

nltk.download('stopwords')
stpwrd = nltk.corpus.stopwords.words('english')

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('accuratee.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
# allow all origins
CORS(app, origins='*')

@app.route('/', methods=['GET'])
def home():
    return 'Hello, World!'


@app.route('/classify', methods=['POST'])
def classify_complaint():
    # Get the complaint text from the form data
    complaint_text = request.form.get('complaint')

    words=[]
    doc = nlp(complaint_text)
    
    for token in doc:
        if str(token) not in stpwrd:
            words.append(token.lemma_)

    corpus = (' '.join(words))

    # vectorizer = TfidfVectorizer()
    E = vectorizer.transform([corpus])

    # Pass the complaint text to the model for classification
    prediction = model.predict(E)

    # Return the classification result
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run()
