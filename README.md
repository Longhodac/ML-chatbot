# ML Chatbot

This is a simple chatbot built using a neural network. The chatbot learns to classify user messages into different categories (called intents) and responds with appropriate answers. I built this project to understand how machine learning can be applied to natural language processing.

## What This Project Does

The chatbot takes a message from the user, figures out what the user is trying to say (greeting, asking for help, saying goodbye, etc.), and then picks a relevant response. It does not generate new sentences. Instead, it matches the user input to a predefined category and returns one of the responses associated with that category.

For example, if the user types "Hi there", the model recognizes this as a greeting and might respond with "Hello" or "Hi there".

## Project Structure

```
ML-chatbot/
├── data/
│   └── intents.json          # Training data with patterns and responses
├── templates/
│   └── index.html            # Web interface for the chatbot
├── static/
│   └── style.css             # Styling for the web interface
├── main.ipynb                # Jupyter notebook for training the model
├── app.py                    # Command line chatbot
├── flask_app.py              # Web application version
├── chat_model.keras          # The trained neural network model
├── tokenizer.pickle          # Saved tokenizer for text processing
├── label_encoder.pickle      # Saved label encoder for categories
└── requirements.txt          # Python dependencies
```

## How the Training Works

The training process happens in the Jupyter notebook (main.ipynb). Here is what each step does:

### 1. Loading the Data

The training data is stored in a JSON file called intents.json. This file contains a list of intents. Each intent has a tag (the category name), a list of patterns (example sentences that belong to this category), and a list of responses (what the chatbot should say when it detects this intent).

```python
with open('data/intents.json') as file:
    data = json.load(file)
```

The code loops through each intent and extracts the patterns as training sentences and the tags as training labels. This gives us pairs of input sentences and their corresponding categories.

### 2. Label Encoding

Neural networks work with numbers, not text. The label encoder converts the category names (like "greeting", "goodbye", "thanks") into integers. So "greeting" might become 0, "goodbye" becomes 1, and so on.

```python
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)
```

We save this encoder later because we need to convert the model output (which will be numbers) back into category names when the chatbot is running.

### 3. Tokenization

Tokenization converts sentences into sequences of numbers. Each unique word gets assigned a unique integer.

```python
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
```

The tokenizer scans all the training sentences and builds a dictionary that maps words to integers. For example, "hello" might be mapped to 5, "help" might be mapped to 12, and so on.

The vocab_size parameter limits how many words we keep. Setting it to 1000 means we only track the 1000 most common words. Any word not in this vocabulary gets replaced with a special token called OOV (out of vocabulary).

### 4. Padding

Neural networks require all inputs to be the same length. But sentences have different lengths. Padding solves this by making all sequences exactly 20 tokens long.

```python
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
```

If a sentence is shorter than 20 tokens, zeros are added at the end. If it is longer, the extra tokens at the end are cut off.

### 5. Building the Model

The model is a sequential neural network with the following layers:

```python
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
```

**Embedding Layer**: This layer turns each word (represented as an integer) into a vector of 16 numbers. Words that appear in similar contexts will have similar vectors. This helps the model understand that "hi" and "hello" are related.

**GlobalAveragePooling1D**: This layer takes the average of all the word vectors in a sentence. This compresses the variable-length sequence into a single fixed-length vector that represents the whole sentence.

**Dense Layers**: These are standard neural network layers. They learn patterns in the data. The relu activation function allows the network to learn non-linear relationships.

**Output Layer**: The final layer has one neuron for each category. The softmax activation converts the outputs into probabilities that sum to 1. The category with the highest probability is the predicted intent.

### 6. Training

```python
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(padded_sequences, np.array(training_labels), epochs=500)
```

The model is trained for 500 epochs. One epoch means the model goes through the entire training dataset once. During training, the model makes predictions, compares them to the correct answers, calculates the error (loss), and adjusts its weights to reduce the error.

The adam optimizer handles the weight updates. The sparse_categorical_crossentropy loss function is used because we have multiple categories and our labels are integers.

### 7. Saving the Model

After training, we save three things:

```python
model.save("chat_model.keras")
```

The model file contains the trained neural network weights. The tokenizer pickle file contains the word-to-integer mapping. The label encoder pickle file contains the integer-to-category mapping. All three are needed when running the chatbot.

## How the Chatbot Works

When the chatbot receives a message, it goes through these steps:

1. The tokenizer converts the message into a sequence of integers
2. The sequence is padded to length 20
3. The model predicts probabilities for each category
4. The category with the highest probability is selected
5. The label encoder converts the category number back to a tag name
6. A random response from that intent is returned

This is the relevant code from flask_app.py:

```python
result = model.predict(keras.preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences([user_message]),
    truncating='post',
    maxlen=max_len
))
tag = lbl_encoder.inverse_transform([np.argmax(result)])
```

## Running the Project

### Requirements

Install the dependencies:

```
pip install -r requirements.txt
```

### Training the Model

Open main.ipynb in Jupyter and run all the cells. This will create the chat_model.keras, tokenizer.pickle, and label_encoder.pickle files.

### Running the Command Line Chatbot

```
python app.py
```

Type messages and press enter. Type "quit" to exit.

### Running the Web Interface

```
python flask_app.py
```

Open your browser and go to http://127.0.0.1:5000

## Adding More Intents

To make the chatbot smarter, edit the data/intents.json file. Add new intents with patterns and responses, then retrain the model by running the notebook again.

Example of adding a new intent:

```json
{
  "tag": "weather",
  "patterns": [
    "What is the weather like",
    "Is it going to rain",
    "How is the weather today"
  ],
  "responses": [
    "I cannot check the weather, but you can look outside",
    "I do not have access to weather data"
  ]
}
```

## Limitations

This chatbot has some limitations:

- It can only respond to topics it was trained on
- It does not understand context or remember previous messages
- It does not generate new text, it only picks from predefined responses
- With limited training data, it may misclassify some messages

## Technologies Used

- Python
- TensorFlow and Keras for the neural network
- scikit-learn for label encoding
- Flask for the web interface
- NumPy for numerical operations
