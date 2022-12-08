import csv
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
from nltk.corpus import stopwords
import numpy as np
import streamlit as st
import pickle
from nltk.stem import WordNetLemmatizer
from PIL import Image

lemma = WordNetLemmatizer()

model = tf.keras.models.load_model('D:/movie-analysis-streamlit/data/model')


def get_sentiment_prediction():
    model = tf.keras.models.load_model(
        'D:/movie-analysis-streamlit/data/model')
    st.write("""
	## Enter a Brief Movie Review
	""")
    review = st.text_area("Enter the review of any movie.", height=100)
    header = ['review']
    data = [review]

    with open('D:/movie-analysis-streamlit/data/predict.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)

    data_review = pd.read_csv('D:/movie-analysis-streamlit/data/predict.csv')

    data_review = preprocessing(data_review)

    vocab_size = 3000
    oov_tok = ''
    max_length = 200

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    X_review = data_review['review'].values
    X_review = tokenizer.texts_to_sequences(X_review)
    X_review_padded = pad_sequences(
        X_review, padding='post', maxlen=max_length, dtype='int32')

    pred_sentiment = model.predict(X_review_padded)

    if pred_sentiment[0] > 0.5:
        st.subheader("Positive Review")
    else:
        st.subheader("Negative Review")
    
    image = Image.open('D:/movie-analysis-streamlit/images/training.jpg')
    st.image(image, caption='Training')
    st.write("Acuuracy over the test set is : 86.6%" )


def preprocessing(data):
    data['review'] = data['review'].apply(lambda x: str(x).replace(
        '<br />', '').replace('\n', '').replace('\'', ''))
    data['review'] = data['review'].apply(
        lambda x: re.sub('[^a-zA-z\s]', '', x))
    data['review'] = data['review'].apply(lambda x: x.lower())

    stop_words = set(stopwords.words("english"))
    data['review'] = data['review'].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in (stop_words)]))

    data['review'] = data['review'].apply(lambda x: ' '.join(
        [lemma.lemmatize(word) for word in x.split()]))

    return data


def main():
    get_sentiment_prediction()


if __name__ == '__main__':
    main()
