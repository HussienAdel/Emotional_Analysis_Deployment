import numpy as np
from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
import h5py

# Initialize Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Download NLTK stopwords
nltk.download('stopwords')

# Verify h5py installation
try:
    logging.info("h5py version: %s", h5py.__version__)
except Exception as e:
    logging.error("h5py not installed: %s", str(e))
    raise e

# Load the model with the custom initializer
try:
    model = load_model('my_model.h5', custom_objects={'Orthogonal': Orthogonal})
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error("Failed to load model: %s", str(e))
    raise e

# Load the vectorizer
try:
    vectorizer = pickle.load(open('vactorizer.pkl', 'rb'))
    logging.info("Vectorizer loaded successfully")
except Exception as e:
    logging.error("Failed to load vectorizer: %s", str(e))
    raise e

# Define emotion labels
labels = {0: "sad", 1: "joy", 2: "love", 3: "angry", 4: "fear", 5: "surprise"}

def preprocess_text(text):
    logging.debug("Starting text preprocessing")
    text = re.sub('[^a-zA-Z]', ' ', text)
    tokens = nltk.word_tokenize(text)
    logging.debug("Tokenized text: %s", tokens)
    tokens = [word.lower() for word in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    logging.debug("Preprocessed text: %s", preprocessed_text)
    return preprocessed_text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.debug("Received data: %s", data)
        if 'text' not in data:
            logging.error("Invalid input: Missing 'text' field")
            return jsonify({'error': 'Invalid input: Missing "text" field'}), 400

        preprocessed_text = preprocess_text(data['text'])
        splited_data = preprocessed_text.split()
        logging.debug("Splitted data: %s", splited_data)

        sequences = vectorizer.texts_to_sequences([splited_data])
        logging.debug("Text sequences: %s", sequences)

        padded_sequences = pad_sequences(sequences, maxlen=80, padding='post')
        logging.debug("Padded sequences: %s", padded_sequences)

        prediction = model.predict(np.array(padded_sequences)).tolist()
        logging.debug("Prediction probabilities: %s", prediction)

        indexed_predictions = [(i, pred) for i, pred in enumerate(prediction[0])]
        indexed_predictions.sort(key=lambda x: x[1], reverse=True)
        out = labels[indexed_predictions[0][0]]
        logging.info("Predicted emotion: %s", out)

        return jsonify({'prediction': out})

    except Exception as e:
        logging.error("Prediction error: %s", str(e))
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == "__main__":
    app.run(debug=True)
