import uvicorn
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import csv
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import Parallel, delayed
import joblib
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
import numpy as np

from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

class Sentiment(BaseModel):
	review: str

app = FastAPI()

model = tf.keras.models.load_model('/sentiment analysis/data/model')

@app.post('/prediction')

def get_sentiment_prediction(review:Sentiment):
	header = ['review']

	with open('/sentiment analysis/data/predict.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow(header)
		writer.writerow(review)

	data_review = pd.read_csv('/sentiment analysis/data/predict.csv')

	data_review = preprocessing(data_review)

	vocab_size = 3000
	oov_tok = ''
	embedding_dim = 100
	max_length = 200
	trunc_type = 'post'
	padding_type = 'post'

	# X_train = np.loadtxt('/sentiment analysis/data/temp.txt', usecols=0,converters=None, dtype = 'str')
	# X_train.reshape((49999,))
	# X_train = pd.read_csv('file.csv')
	tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
	tokenizer.fit_on_texts(X_train)
	X_review = data_review['review'].values
	X_review = tokenizer.texts_to_sequences(X_review)
	X_review_padded =  pad_sequences(X_review, padding='post', maxlen=max_length, dtype = 'int32')
	
	pred_sentiment = model.predict(X_review_padded)

	if pred_sentiment[0] > 0.55:
		return {'prediction': 'positive'}
	elif pred_sentiment[0] < 0.45:
		return {'prediction': 'negative'}
	return {'prediction': 'neutral'}

data=pd.read_csv('/sentiment analysis/data/IMDB Dataset.csv')
	
def preprocessing(data):
	data['review'] = data['review'].apply(lambda x: x.replace('<br />', '').replace('\n', '').replace('\'', ''))
	data['review'] = data['review'].apply(lambda x: re.sub('[^a-zA-z\s]','',x))
	data['review'] = data['review'].apply(lambda x:x.lower())

	stop_words = set(stopwords.words("english"))
	data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

	data['review']  = data['review'].apply(lambda x: ' '.join([lemma.lemmatize(word) for word in x.split()]))

	return data

data = preprocessing(data)
reviews = data['review'].values
labels = data['sentiment'].values
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(reviews, encoded_labels,test_size = 0.00001)

# pd.DataFrame({'text':X_train}).to_csv('file.csv', index = False)
# print(type(X_train))
# np.savetxt('temp.txt', X_train, fmt='%s')