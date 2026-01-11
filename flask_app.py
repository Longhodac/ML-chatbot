from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random
import pickle

app = Flask(__name__)

# Load data and model at startup
with open("data/intents.json") as file:
    data = json.load(file)

model = keras.models.load_model('chat_model.keras')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

max_len = 20


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')

    if not user_message:
        return jsonify({'response': 'Please enter a message'})

    # Predict intent
    result = model.predict(keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences([user_message]),
        truncating='post',
        maxlen=max_len
    ))

    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    # Find matching intent and get response
    for intent in data['intents']:
        if intent['tag'] == tag:
            bot_response = np.random.choice(intent['responses'])
            return jsonify({'response': bot_response})

    return jsonify({'response': "I'm not sure I understand. Can you rephrase that?"})


if __name__ == '__main__':
    app.run(debug=True)
