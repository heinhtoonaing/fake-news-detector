import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import numpy as np

# Load the saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Detector")
st.subheader("Check if a news article is REAL or FAKE")

user_input = st.text_area("Paste the news article or headline here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)[0]

        # Get decision function score and turn it into confidence
        score = model.decision_function(vectorized_input)
        confidence = np.max(np.exp(score) / np.sum(np.exp(score)))
        confidence_percent = round(confidence * 100, 2)

        # Show result with emoji + confidence
        if prediction == "FAKE":
            st.error(f"🚨 This news is likely **FAKE**! ({confidence_percent}% confident)")
        else:
            st.success(f"✅ This news is likely **REAL**. ({confidence_percent}% confident)")
