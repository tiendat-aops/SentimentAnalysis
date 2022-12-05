import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import joblib
from pathlib import Path
import re
import nltk
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
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
from joblib import Parallel, delayed
import joblib

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

	return X_train_padded, y_train


def training(data):
	X_train_padded, y_train = preprocessing(data)
	# data = preprocessing(data)

	# wc = ' '.join([text for text in data['review'][data['sentiment']=='positive']])

	# reviews = data['review'].values
	# labels = data['sentiment'].values
	# encoder = LabelEncoder()
	# encoded_labels = encoder.fit_transform(labels)

	# X_train, X_test, y_train, y_test = train_test_split(reviews, encoded_labels,test_size = 0.25)

	vocab_size = 3000
	oov_tok = ''
	embedding_dim = 100
	max_length = 200
	trunc_type = 'post'
	padding_type = 'post'

	# tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
	# tokenizer.fit_on_texts(X_train)
	# word_index = tokenizer.word_index

	# X_train = tokenizer.texts_to_sequences(X_train)
	# X_train_padded = pad_sequences(X_train, padding='post', maxlen=max_length)

	# X_test = tokenizer.texts_to_sequences(X_test)
	# X_test_padded = pad_sequences(X_test, padding='post', maxlen=max_length)

	model = Sequential()
	model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
	model.add(Bidirectional(LSTM(128)))
	# model.add(LSTM(128))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dense(1, activation = 'sigmoid'))

	model.summary()

	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

	with tf.device("/gpu:0"):
		# history = model.fit(X_train_padded, y_train, epochs=1, verbose=1,steps_per_epoch = len(X_train)/512,batch_size = 32,validation_split=0.1)
		history = model.fit(X_train_padded, y_train, epochs=10, verbose=1,steps_per_epoch = 146,batch_size = 128,validation_split=0.1)

	# pkl_filename = '/sentiment analysis/data/model.pkl'
	# with open(pkl_filename, 'wb') as f:
	# 	pickle.dump(model, f)
	# tf.saved_model.save(model, '/sentiment analysis/data/model.txt')
	# joblib.dump(model, '/sentiment analysis/data/model.pkl')
	model.save('/sentiment analysis/data/model')


training(data)

# import csv

# X_train_padded, y_train, tokenizer = preprocessing(data)

# def get_sentiment_prediction(data, tokenizer):
# 	header = ['review']

# 	with open('/sentiment analysis/data/predict.csv', 'w') as f:
# 		writer = csv.writer(f)
# 		writer.writerow(header)
# 		writer.writerows(data)

# 	data_review = pd.read_csv('/sentiment analysis/data/predict.csv')

# 	data_review['review'] = data_review['review'].apply(lambda x: x.replace('<br />', '').replace('\n', '').replace('\'', ''))
# 	data_review['review'] = data_review['review'].apply(lambda x: re.sub('[^a-zA-z\s]','',x))
# 	data_review['review'] = data_review['review'].apply(lambda x:x.lower())
# 	stop_words = set(stopwords.words("english"))
# 	data_review['review'] = data_review['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
# 	data_review['review']  = data_review['review'].apply(lambda x: ' '.join([lemma.lemmatize(word) for word in x.split()]))

# 	vocab_size = 3000
# 	oov_tok = ''
# 	embedding_dim = 100
# 	max_length = 200
# 	trunc_type = 'post'
# 	padding_type = 'post'

# 	X_review = data_review['review'].values
# 	print(X_review)
# 	X_review = tokenizer.texts_to_sequences(X_review)
# 	X_review_padded =  pad_sequences(X_review, padding='post', maxlen=max_length, dtype = 'int32')
# 	print(X_review_padded)
# 	pred_sentiment = model.predict(X_review_padded)

# 	print('true')

# model = tf.keras.models.load_model('/sentiment analysis/data/saved_model')

# get_sentiment_prediction('this movie is so bad', tokenizer)