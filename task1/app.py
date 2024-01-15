import nltk
import spacy
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model for named entity recognition
nlp = spacy.load("en_core_web_sm")

# Sample dataset with intents
dataset = [
    ("Hi there!", "greeting"),
    ("How can I help you?", "greeting"),
    ("What's the latest in computer graphics?", "comp.graphics"),
    ("Tell me about sports", "rec.sport.baseball"),
    ("I'm feeling sick", "sci.med"),
]

# Text processing using spaCy for named entity recognition
def process_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    # Tokenization and named entity recognition
    doc = nlp(text)
    words = [ps.stem(token.text) for token in doc if token.text.lower() not in stop_words]

    return ' '.join(words)

# Intent recognition using a simple neural network with TensorFlow
def create_model(input_size, output_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, input_dim=input_size, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# One-hot encode intents
intents = list(set(intent for _, intent in dataset))
intent_to_index = {intent: i for i, intent in enumerate(intents)}
index_to_intent = {i: intent for i, intent in enumerate(intents)}

# Prepare training data
processed_data = [(process_text(text), intent_to_index[intent]) for text, intent in dataset]
X_train, y_train = zip(*processed_data)

# Create a Bag-of-Words representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)

# Split data into training and validation sets
X_train_bow, X_val_bow, y_train, y_val = train_test_split(X_train_bow, y_train, test_size=0.2, random_state=42)

# Train the intent recognition model
model = create_model(X_train_bow.shape[1], len(intents))
y_train_one_hot = tf.keras.utils.to_categorical(y_train, len(intents))

model.fit(X_train_bow.toarray(), y_train_one_hot, epochs=5, batch_size=8, validation_data=(X_val_bow.toarray(), tf.keras.utils.to_categorical(y_val, len(intents))))

# Chatbot loop
while True:
    user_input = input("User: ")

    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break

    user_input_processed = process_text(user_input)
    user_input_bow = vectorizer.transform([user_input_processed])

    # Predict intent
    predicted_intent_index = np.argmax(model.predict(user_input_bow.toarray()))
    predicted_intent = index_to_intent[predicted_intent_index]

    # Generate response
    response = f"I'm sorry, I don't know how to respond to that."  # You can replace this with a more elaborate response generation mechanism.

    print(f"Chatbot: {response}")
