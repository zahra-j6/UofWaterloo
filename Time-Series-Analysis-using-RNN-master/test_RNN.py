# import required packages
import pandas as pd
import numpy as np
from os import path

from keras.engine.saving import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from math import sqrt
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib.pyplot as plt


if __name__ == "__main__":
    np.random.seed(42)

    # 1. load your training data
    df=read_csv('test_data_RNN.csv', header=0, index_col=0)
  #  dataset = read_csv('train_data_RNN.csv', header=0, index_col=0)
    dataset = df.drop('Date', axis=1)  # dropping date column
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]  # dropped first column
    #print(dataset)
    #training_set = dataset.iloc[:,:12].values
    #testing_set= dataset.iloc[:,12:].values
    #print(training_set)

    # loading the same normalization used in training
    with open('scaler.pickle', 'rb') as handle:
        sc = pickle.load(handle)
    #inputs=dataset['Target']
    inputs = dataset.values.reshape(-1, 13)

    #Normalizing the test data
    inputs = sc.transform(inputs)

  #  sc = MinMaxScaler(feature_range=(0,1))
#    training_set_scaled = sc.transform(dataset)

    X_test=inputs[:,:12]
    #reshaping the test data to make it 3D
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    y_test=inputs[:,12:]

    # 2. Load your testing data
    model=load_model('20858708_RNN_model.h5')
    # 3. Run prediction on the test data and output required plot and loss
    print(model.evaluate(X_test,y_test))
    predicted_stock_price = model.predict(X_test)
    print(predicted_stock_price.shape)
    print(type(predicted_stock_price))

    # create empty table with 12 fields
    trainPredict_dataset_like = np.zeros(shape=(len(dataset), 13) )
    # put the predicted values in the right field
    trainPredict_dataset_like[:,0] = predicted_stock_price[:,0]
    # inverse transform and then select the right field
    trainPredict = sc.inverse_transform(trainPredict_dataset_like)[:,0]
   # print(trainPredict)
    trainPredict_dataset_like2 = np.zeros(shape=(len(dataset), 13))
    trainPredict_dataset_like2[:,0] = y_test[:,0]
    y = sc.inverse_transform(trainPredict_dataset_like2)[:, 0]


    #sorted_ind = np.argsort(y)
    plt.plot(y, label='Predicted Values')  # invert scaling for forecast
    plt.plot(trainPredict , label='Actual values')  # actual
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    plt.savefig('RNN_Market.png')

    sorted_ind=np.argsort(y)
    plt.plot(y[sorted_ind], label='Predicted Values')  # invert scaling for forecast
    plt.plot(trainPredict[sorted_ind], label='Actual values')  # actual
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
