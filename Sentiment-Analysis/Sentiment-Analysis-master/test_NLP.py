# import required packages
import pandas as pd
from utils import *
from keras.models import load_model
from os import path
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
import pickle

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
	# 1. Load your saved model
    loaded_model = load_model('20858708_NLP_model.h5')
    print(loaded_model.summary())
	# 2. Load your testing data
    if path.exists("imdb_test3.csv"):
        pass
    else:
        preparing('imdb_test3.csv', False)
    data = pd.read_csv('imdb_test3.csv')



    X = data['text']
    y = data['rating']
    # loading the tokenizer, which we have saved during the training step.
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    test_review_tokenized = tokenizer.texts_to_sequences(X)
    x_test = pad_sequences(test_review_tokenized, maxlen=500)

   # tokenizer = Tokenizer(num_words=5000)  # try 6000
   # tokenizer.fit_on_texts(X)
   # X=tokenizer.texts_to_sequences(X)
   # X = pad_sequences(X,  maxlen=500)



#    X_token = tokenizer.texts_to_sequences(X)
 #   X_test = pad_sequences(X_token, maxlen=100)
    # 3. Run prediction on the test data and print the test accuracy
    prediction = loaded_model.predict(x_test)
    y_pred = (prediction > 0.5)

    print('Accuracy is',(accuracy_score(y_pred, y))*100)


