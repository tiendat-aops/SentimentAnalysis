import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from nltk.corpus import stopwords
from textblob import Word
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
import csv
data=pd.read_csv('/sentiment analysis/data/IMDB Dataset.csv')

from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

def preprocessing(data):
	data['review'] = data['review'].apply(lambda x: x.replace('<br />', '').replace('\n', '').replace('\'', ''))
	data['review'] = data['review'].apply(lambda x: re.sub('[^a-zA-z\s]','',x))
	data['review'] = data['review'].apply(lambda x:x.lower())

	stop_words = set(stopwords.words("english"))
	data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

	data['review']  = data['review'].apply(lambda x: ' '.join([lemma.lemmatize(word) for word in x.split()]))

	reviews = data['review'].values
	labels = data['sentiment'].values
	encoder = LabelEncoder()
	encoded_labels = encoder.fit_transform(labels)

	X_train, X_test, y_train, y_test = train_test_split(reviews, encoded_labels,test_size = 0.00001)

	X = X_train
	vocab_size = 3000
	oov_tok = ''
	embedding_dim = 100
	max_length = 200
	trunc_type = 'post'
	padding_type = 'post'

	tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
	tokenizer.fit_on_texts(X_train)
	word_index = tokenizer.word_index

	X_train = tokenizer.texts_to_sequences(X_train)
	X_train_padded = pad_sequences(X_train, padding='post', maxlen=max_length)

	# X_test = tokenizer.texts_to_sequences(X_test)
	# X_test_padded = pad_sequences(X_test, padding='post', maxlen=max_length)

	return X,X_train_padded, y_train


def training(data):
	X_train, X_train_padded, y_train = preprocessing(data)

	vocab_size = 3000
	embedding_dim = 100
	max_length = 200

	model = Sequential()
	model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
	model.add(Bidirectional(LSTM(128)))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dense(1, activation = 'sigmoid'))

	model.summary()

	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

	with tf.device("/gpu:0"):
		history = model.fit(X_train_padded, y_train, epochs=1, verbose=1,steps_per_epoch = len(X_train)/512,batch_size = 32,validation_split=0.1)

	model.save('D:/movie-analysis-streamlit/data/model')
