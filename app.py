
import numpy as np
from flask import Flask, request, jsonify, render_template
import os
import pickle
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    tokens = nltk.word_tokenize(text)

    # Convert all the words to lowercase
    tokens = [word.lower() for word in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stem the words
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


# Load the model with the custom initializer
model = load_model('my_model.h5', custom_objects={'Orthogonal': Orthogonal})

vectorizer = pickle.load(open('vactorizer.pkl', 'rb'))

# Verify the model by printing its summary
#loaded_model.summary()

app = Flask(__name__) # Initialize the flask App

labels = {0:"sad", 1:"joy", 2:"love", 3:"angry", 4:"fear", 5:"surprise"}

@app.route('/predict', methods=['POST'])
def predict():
    
    # Get data from the request
    data = request.get_json()

    # Preprocess the text
    preprocessed_text = preprocess_text(data['text'])
    
    splited_data = preprocessed_text.split()

    # Vectorize the preprocessed text
    sequences = vectorizer.texts_to_sequences([splited_data])
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=80, padding='post')

    # Make a prediction
    prediction = model.predict(np.array(padded_sequences)).tolist()
    
    indexed_predictions = [(i, pred) for i, pred in enumerate(prediction[0])]
    
    indexed_predictions.sort(key=lambda x: x[1], reverse=True)

    out = labels[indexed_predictions[0][0]]

    # Return the predictions as JSON
    return jsonify({'prediction': out})


if __name__ == "__main__":
    app.run(debug=True)

