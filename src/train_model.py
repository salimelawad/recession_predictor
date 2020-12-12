import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from numpy import array
from numpy import hstack

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from seaborn import lineplot

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import data_ingestor

api_key = '6dcd215d39286437b67f6389b88903e0'

def generate_future_rec_counts(usrec, months_ahead=9, month_group=6):
    return usrec.shift(-1 * months_ahead)\
                .iloc[::-1]\
                .rolling(month_group, min_periods=0)\
                .sum()\
                .iloc[::-1]
                
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def scale(X):
    # Scale features
    scaler = MinMaxScaler()
    scaler.fit(X)
    scaled_features = scaler.transform(X)
    return scaled_features, scaler
    
def create_model(nodes, activation, timesteps, n_features, dropout, optimizer, loss):
    model = Sequential()
    model.add(LSTM(nodes, activation=activation, input_shape=(timesteps, n_features), dropout=dropout))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss)
    
    return model
    
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, epochs):
  
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    
    pred = model.predict(X_test)
    
    return mean_squared_error(y_test, pred)
    
#fit_cross_val(train, test, 10, 'relu', .1, 'adam', 'mse' 200)

def cross_val(np_data, timesteps, n_folds, nodes, activation, dropout, optimizer, loss, epochs):
    scores = []
    X, y = load_data(np_data, timesteps)
    skf = KFold(n_splits=n_folds, random_state=None, shuffle=False)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print ("Running Fold", i+1, "/", n_folds, end='')
        model = None # Clearing the NN.
        model = create_model(nodes, activation, timesteps, 4, dropout, optimizer, loss)
        score = train_and_evaluate_model(model, X[train_index], y[train_index], X[test_index], y[test_index], epochs)
        scores.append(score)
        print(f", score = {score}")
        
    return mean(scores)

def graph_results(index, predictions, scaled_target):
    fig, ax = plt.subplots(figsize=(25,12)) 
    #myFmt = mdates.DateFormatter("%y-%m")
    #ax.xaxis.set_major_formatter(myFmt)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax.set_title('Preds', size= 30)
#    wide_df = data[['preds', target_col]]
    wide_df = pd.DataFrame(index=index)
    wide_df['predictions'] = predictions
    wide_df['target'] = scaled_target
    ax = lineplot(data=wide_df)
    plt.xlabel('Year', size=20)
    plt.ylabel('# of future months in recession', size=20)
    plt.xticks(rotation=45)
    plt.grid(which='major')

data = data_ingestor.load_daily(api_key)
usrec = data_ingestor.load_recession(api_key)
data['rec_group'] = generate_future_rec_counts(usrec, months_ahead=180, month_group=180).fillna('bfill').fillna('ffill')
dataset = data.reset_index()[['T3MFF', 'T1YFF', 'T5YFF', 'T10YFF', 'rec_group']].to_numpy()
time_index = data.index.to_numpy()

# choose a number of time steps
n_steps = 6
# convert into input/output
X, y = split_sequences(dataset, n_steps)
time_steps_crop = time_index[n_steps-1:]
X_train = X[:-19]
y_train = y[:-19]
ts_train = time_steps_crop[0:514]
X_test = X[514:]
y_test = y[514:]
ts_test = time_steps_crop[514:]
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

model = create_model(nodes=100, activation='relu', timesteps=n_steps, n_features=4, dropout=.3, optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=200, verbose=0)
pred_t = model.predict(X)

graph_results(hstack(time_steps_crop), y, pred_t)