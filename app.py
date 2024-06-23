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

# Initialize Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG

# Download NLTK stopwords
nltk.download('stopwords')

# Load the model with the custom initializer
try:
    model = load_model('my_model.h5', custom_objects={'Orthogonal': Orthogonal})
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error("Failed to load model: %s", str(e))
    raise e  # Raise exception to terminate application if model loading fails

# Load the vectorizer
try:
    vectorizer = pickle.load(open('vactorizer.pkl', 'rb'))
    logging.info("Vectorizer loaded successfully")
except Exception as e:
    logging.error("Failed to load vectorizer: %s", str(e))
    raise e  # Raise exception to terminate application if vectorizer loading fails

# Define emotion labels
labels = {0: "sad", 1: "joy", 2: "love", 3: "angry", 4: "fear", 5: "surprise"}

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
    
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()
        
        # Log the received data
        logging.debug("Received data: %s", data)
        
        # Validate input data
        if 'text' not in data:
            return jsonify({'error': 'Invalid input: Missing "text" field'}), 400
        
        # Preprocess the text
        preprocessed_text = preprocess_text(data['text'])
        
        # Split the preprocessed text into tokens
        splited_data = preprocessed_text.split()

        # Vectorize the preprocessed text
        sequences = vectorizer.texts_to_sequences([splited_data])
        
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=80, padding='post')

        # Make a prediction
        prediction = model.predict(np.array(padded_sequences)).tolist()
        
        # Log the prediction
        logging.debug("Prediction probabilities: %s", prediction)
        
        # Get the index of the highest probability
        indexed_predictions = [(i, pred) for i, pred in enumerate(prediction[0])]
        indexed_predictions.sort(key=lambda x: x[1], reverse=True)
        out = labels[indexed_predictions[0][0]]
        
        # Log the final prediction
        logging.info("Predicted emotion: %s", out)

        # Return the prediction as JSON response
        return jsonify({'prediction': out})
    
    except Exception as e:
        # Log any exceptions that occur during prediction
        logging.error("Prediction error: %s", str(e))
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == "__main__":
    app.run(debug=True)
