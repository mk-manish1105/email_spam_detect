import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load the model
with open("email_model_pkl", "rb") as f:
    model = pickle.load(f)

# Function to predict spam
def predict_spam(email_message):
    prediction = model.predict([email_message])
    return prediction[0]

def main():
    st.title("Welcome to Manish Email Spam Detection Machine Learning Model")
    st.title("Email Spam Detection")
    st.write("This web app predicts whether an email message is spam or not.")

    email_message = st.text_area("Enter your email message here:", "")

    if st.button("Predict"):
        if email_message.strip() == "":
            st.error("Please enter an email message.")
        else:
            prediction = predict_spam(email_message)
            if prediction == 1:
                st.success("This email is likely spam.")
            else:
                st.success("This email is not spam.")

if __name__ == "__main__":
    main()
