# import required packages
import pandas as pd
import numpy as np
from os import path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.models import Sequential
from pandas import read_csv
import pickle

def prepare():
    df = pd.read_csv('q2_dataset.csv') #import csv
    date = []
    date = df['Date']
    vol = df[' Volume']
    op = df[' Open']
    high = df[' High']
    low = df[' Low']
    day1_vol = []
    day1_high = []
    day1_low = []
    day1_open = []
    day2_vol = []
    day2_high = []
    day2_low = []
    day2_open = []

    day3_vol = []
    day3_high = []
    day3_low = []
    day3_open = []

    target = []
    #  target.append(' ')

    j = 0
    for i in range(0, len(df) - 3):
        j = 0
        for j in range(0, 4): #current date's open value is the target to predict
            if j == 0:
                #              if i+j-1>-1:
                target.append(op[i])
            #             else:
            #                 target.append(0)
            # j=j+1
            # for the current date the previous day's volume, low and high value are stored as a feature
            if j == 1:
                day1_vol.append(vol[i + j])
                day1_high.append(high[i + j])
                day1_low.append(low[i + j])
                day1_open.append(op[i + j])

                # for the current date the previous to one day's volume, low and high value are stored as a feature
            # j=j+1
            if j == 2:
                day2_vol.append(vol[i + j])
                day2_high.append(high[i + j])
                day2_low.append(low[i + j])
                day2_open.append(op[i + j])

                # j=j+1
                # for the current date the previous to 2 day's volume, low and high value are stored as a feature

            if j == 3:
                day3_vol.append(vol[i + j])
                day3_high.append(high[i + j])
                day3_low.append(low[i + j])
                day3_open.append(op[i + j])

                # j=j-1

    Dataset = list(
        zip(date, day1_open, day1_high, day1_low, day1_vol, day2_open, day2_high, day2_low, day2_vol, day3_open,
            day3_high, day3_low, day3_vol, target))
    print(Dataset)
    np.random.shuffle(Dataset) # randomize the created data

    Database = pd.DataFrame(data=Dataset,
                            columns=['Date', 'Day 1 Open', 'Day 1 High', 'Day 1 Low ', 'Day 1 Volume', 'Day 2 Open',
                                     'Day 2 High', 'Day 2 Low ', 'Day 2 Volume', 'Day 3 Open', 'Day 3 High',
                                     'Day 3 Low ', 'Day 3 Volume', 'Target'])
    train = Database.values[0:879, :] # splitting
    test = Database.values[879:, :]
    # training_set=Dataset[0:4,:]
    print('Training set is', train)
    #creating train_data_RNN.csv
    pd.DataFrame(train,
                 columns=['Date', 'Day 1 Open', 'Day 1 High', 'Day 1 Low ', 'Day 1 Volume', 'Day 2 Open', 'Day 2 High',
                          'Day 2 Low ', 'Day 2 Volume', 'Day 3 Open', 'Day 3 High', 'Day 3 Low ', 'Day 3 Volume',
                          'Target']).to_csv("train_data_RNN.csv")  # target to remove from here
    print('Testing set is ', test)
    #creating test_data_RNN.csv

    pd.DataFrame(test, columns=['Date', 'Day 1 Open', 'Day 1 High','Day 1 Low ','Day 1 Volume','Day 2 Open', 'Day 2 High','Day 2 Low ','Day 2 Volume','Day 3 Open', 'Day 3 High','Day 3 Low ','Day 3 Volume','Target']).to_csv("test_data_RNN.csv")


if __name__ == "__main__":
    if path.exists("train_data_RNN.csv") and path.exists("test_data_RNN.csv"):
        pass
    else:
        print('Preparing data')
        prepare()
    np.random.seed(42)
    # 1. load your training data
    df=read_csv('train_data_RNN.csv', header=0, index_col=0)
  #  dataset = read_csv('train_data_RNN.csv', header=0, index_col=0)
    dataset = df.drop('Date', axis=1)  # dropping date column
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]  # dropped first column
    #print(dataset)
    #training_set = dataset.iloc[:,:12].values
    #testing_set= dataset.iloc[:,12:].values
    #print(training_set)

    #Normalization of entire data
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(dataset)
    #X_train is all columns except the target column
    X_train=training_set_scaled[:,:12]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    print(X_train.shape)
    y_train=training_set_scaled[:,12:]
    print(y_train.shape)
    # 2. Train your network
    model = Sequential()
    model.add(LSTM(units=100,return_sequences=True,input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam',loss='mean_squared_error')
    # 		Make sure to print your training loss within training to show progress
    model.fit(X_train,y_train,epochs=100,batch_size=1)
    # 		Make sure you print the final training loss
    print(model.evaluate(X_train,y_train))
    # 3. Save your model
    model.save('20858708_RNN_model.h5')
    with open('scaler.pickle', 'wb') as handle:
        pickle.dump(sc, handle, protocol=pickle.HIGHEST_PROTOCOL)

