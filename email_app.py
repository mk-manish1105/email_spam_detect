import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset
df = pd.read_csv("E:\\demo file\\spam.csv")

# Preprocess the dataset
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
x_train, x_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.25)
email_model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
email_model.fit(x_train, y_train)

# Save the model using pickle
with open("E:\\CODING\\Python\\Machine Learning\\Machine Learning Project\\Email Spam Detection\\email_model_pickle", "wb") as f:
    pickle.dump(email_model, f)

# Load the saved model
with open("E:\\CODING\\Python\\Machine Learning\\Machine Learning Project\\Email Spam Detection\\email_model_pickle", "rb") as f:
    model = pickle.load(f)

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
