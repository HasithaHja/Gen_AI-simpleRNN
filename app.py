import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Get the word index dictionary
word_index = imdb.get_word_index()
reverse_word_index = {value + 3: key for key, value in word_index.items()}

# Load the model
model = load_model('simple_RNN_imdb.h5')

# Function to decode a review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review])

# Function to preprocess user inputs
def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]
    return sequence.pad_sequences([encoded], maxlen=500)

# Prediction function
def predict_sentiment(review):
    processed = preprocess_text(review)
    prediction = model.predict(processed)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]

# Streamlit app
st.title("IMDB movie review sentiment analysis")
st.write("Enter a movie review to classify it as positive or negative.")

user_input = st.text_area("Movie Review")

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.45 else "Negative"

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Score: {prediction[0][0]}")
else:
    st.write("Please enter a movie review and click 'Classify' to see the sentiment analysis.")


