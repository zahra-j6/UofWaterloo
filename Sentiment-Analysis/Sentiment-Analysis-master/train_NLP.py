# import required packages
import pandas as pd
from utils import *
import os.path
from os import path
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from numpy import zeros
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
from keras.layers import Bidirectional, GlobalMaxPool1D,MaxPool1D
from keras.layers.convolutional import MaxPooling1D,Conv1D
from keras.optimizers import Adam
from keras.layers import LSTM,GRU
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint, History


if __name__ == "__main__":

	# 1. load your training data
    if path.exists("imdb_train3.csv"):
        pass
    else:
        preparing('imdb_train3.csv',False)
    data = pd.read_csv('imdb_train3.csv')
    X = data['text']
    y = data['rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    np.random.seed(42)
    num_most_freq_words_to_include = 5000
    MAX_REVIEW_LENGTH_FOR_KERAS_RNN = 500  # Input for keras.
    embedding_vector_length = 32

    all_review_list = X_train + X_test

    tokenizer = Tokenizer(num_words=num_most_freq_words_to_include)
    tokenizer.fit_on_texts(X_train)

    # tokenisingtrain data
    train_reviews_tokenized = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(train_reviews_tokenized, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)  # 20,000 x 500

    # tokenising validation data
    val_review_tokenized = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(val_review_tokenized, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)  # 5000 X 500

    # tokenising Test data
#    test_review_tokenized = tokenizer.texts_to_sequences(testcleanWords)
 #   x_test = pad_sequences(test_review_tokenized, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)  # 5000 X 500

    # Save the tokenizer, so that we can use this tokenizer whenever we need to predict any reviews.
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    checkpointer = ModelCheckpoint(filepath="keras.model", verbose=1, save_best_only=True)
    monitor = EarlyStopping(monitor='val_accuracy', min_delta=1e-3, patience=10, verbose=1, mode='auto',
                            restore_best_weights=True)

    #   tokenizer.fit_on_texts(X_train)
 #   X_train = tokenizer.texts_to_sequences(X_train)
 #   X_test = tokenizer.texts_to_sequences(X_test)
    # Adding 1 because of reserved 0 index

    #vocab_size = len(tokenizer.word_index) + 1
    '''
    model = Sequential()
    model.add(Embedding(input_dim=num_most_freq_words_to_include,
                        output_dim=embedding_vector_length,
                        input_length=MAX_REVIEW_LENGTH_FOR_KERAS_RNN))

    model.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(GRU(100))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())
    history = model.fit(X_train, y_train, batch_size=64, epochs=3, verbose=2, validation_data=(X_test,y_test))'''
    embedding_size=32
    model=Sequential()
    model.add(Embedding(5000, embedding_size, input_length=500))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=13,callbacks=[monitor,checkpointer])#64,13

    score = model.evaluate(X_test, y_test, verbose=1)
    print("Test loss:", score[0])
    print("Test Accuracy:", score[1])
	# 		Make sure you print the final training accuracy

    prediction = model.predict(X_test)
    y_pred = (prediction > 0.5)

    print('Training Accuracy is', (accuracy_score(y_pred, y_test)) * 100)
    # 3. Save your model
    model_name = '20858708_NLP_model.h5'
    model.save(model_name)
    print("Saved model to disk")