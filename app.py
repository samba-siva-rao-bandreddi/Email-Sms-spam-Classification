import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

snow_stemmer=SnowballStemmer("english")


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(snow_stemmer.stem(i))

    return ' '.join(y)


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title("Email/SMS Spam Classification")
input_sms=st.text_area("Enter the message")
if st.button('Predict'):
    #preprocess
    transformed_sms=transform_text(input_sms)
    #vectorize
    vector_input=tfidf.transform([transformed_sms])
    #predict
    result=model.predict(vector_input)[0]
    #Display
    if result==1:
        st.header("spam")
    else:
        st.header("Not spam")


