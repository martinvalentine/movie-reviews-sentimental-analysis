import streamlit as st
import pickle
import numpy as np
import re

# Load the model and vectorizer
def load_model():
    with open('../Model/model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# Load the saved model and vectorizer
data = load_model()
predictor = data['model']
vectorizer = data['vectorizer']

# Function to clean input text while retaining punctuation (! and ?)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s!?]', '', text)
    return text

# Define the function for the prediction page
def show_predict_page():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="📊", layout="centered")
    st.title('📊 Sentiment Analysis')
    st.write("Analyze the sentiment of your text review. This app predicts whether the sentiment of the input text is positive or negative.")

    # Add a sidebar with instructions
    st.sidebar.header("About the App")
    st.sidebar.write("This application uses a pre-trained model to perform sentiment analysis on text reviews. Enter any text review, and the app will predict if it’s positive or negative.")

    # Input review from the user
    review = st.text_area("Enter your review below:", height=150, placeholder="Type your review here...")

    # Adding a button for prediction
    if st.button("Predict Sentiment"):
        if review:
            # Clean the input text
            cleaned_review = clean_text(review)

            # Transform the cleaned review using the loaded vectorizer
            review_transformed = vectorizer.transform([cleaned_review])

            # Make the prediction using the trained model
            prediction = predictor.predict(review_transformed)

            # Display the result with styling based on sentiment
            sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😞"
            sentiment_color = "green" if prediction[0] == 1 else "red"
            st.markdown(f"<h2 style='color: {sentiment_color};'>Sentiment: {sentiment}</h2>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a review to predict.")
