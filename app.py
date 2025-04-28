# app.py

import streamlit as st
import pickle
import string

# Load model and vectorizer
model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Streamlit Interface
st.title("📩 SMS Spam Classifier")

st.write("Enter an SMS message to classify it as Spam or Not Spam.")

message = st.text_area("Enter your message here:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        processed = preprocess(message)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("🚫 This message is SPAM.")
        else:
            st.success("✅ This message is NOT Spam (Ham).")
