from flask import Flask, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import string
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import spacy
from spacy import displacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.util import filter_spans

app = Flask(__name__)

# Initialize NLTK
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load TensorFlow/Keras model for intent classification
model = load_model('intent_classification_model.h5')

# Load responses
with open('responses.json') as file:
    responses = json.load(file)

# Initialize matcher for SpaCy
matcher = Matcher(nlp.vocab)

# Add patterns for named entity recognition
matcher.add("MENTAL_HEALTH_ISSUE", [[{"LOWER": {"IN": ["depression", "anxiety", "stress", "panic disorder"]}}]])

# Route for receiving messages from the user
@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    message = data["message"]

    # Process the message and generate a response
    response = process_message(message)

    return jsonify({"response": response})

# Function to process the user's message and generate a response
def process_message(message):
    # Tokenize the message
    tokens = nltk.word_tokenize(message)
    # Lemmatize each word
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word not in string.punctuation]

    # Perform intent classification using TensorFlow/Keras model
    intent_probabilities = model.predict([tokens])[0]
    intent_index = np.argmax(intent_probabilities)
    intent = ["greetings", "fallback"][intent_index]

    # Perform named entity recognition using SpaCy
    doc = nlp(message)
    mental_health_entities = [ent.text for ent in doc.ents if ent.label_ == "MENTAL_HEALTH_ISSUE"]

    # If mental health issue entities are found, provide appropriate response
    if mental_health_entities:
        return random.choice(responses["mental_health_issue"])

    # If no specific intent or entities are detected, provide fallback response
    return responses[intent]

if __name__ == '__main__':
    app.run(debug=True)
