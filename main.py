# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from seaborn import lineplot
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


### READ DATA INTO ONE DATAFRAME

a = pd.read_csv('T1YFFM.csv', index_col='DATE')
b = pd.read_csv('T5YFFM.csv', index_col='DATE')
c = pd.read_csv('T10YFFM.csv', index_col='DATE')
d = pd.read_csv('TB3SMFFM.csv', index_col='DATE')

r = pd.read_csv('USREC.csv', index_col='DATE')

data = pd.concat([a, b, c, d, r], axis=1)
data.index = pd.to_datetime(data.index)

data['future_rec'] = data['USREC'].shift(-4).iloc[::-1].rolling(6, min_periods=0).sum().iloc[::-1]

for i in range(24):
    shift = i + 1
    data['USREC_shift_'+str(shift)] = data['USREC'].shift(shift * -1)
    
data = data.fillna(0)
### Split target
feature_col = ['TB3SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM']
target_col = 'future_rec'





def bad_model_test(data, date_filter, feature_cols, target_col, graph=False):
    #filter
    data_filtered = data[data.index < date_filter].copy(deep=True)
    
    features = data_filtered[feature_cols].copy(deep=True)
    target = data_filtered[[target_col]].copy(deep=True)
    
    # Scale features
    scaler_feature = MinMaxScaler()
    scaler_feature.fit(features)
    scaled_features = scaler_feature.transform(features)
    
    # Scale target
    scaler_target = MinMaxScaler()
    scaler_target.fit(target)
    scaled_target = np.ravel(scaler_target.transform(target))
    
    regr = MLPRegressor(hidden_layer_sizes= (100, 100, 100))
    model = regr.fit(scaled_features, scaled_target)
    predictions = model.predict(scaled_features)
    data_filtered['preds'] = predictions
    print(mse(data_filtered['preds'], scaled_target))
    if graph:
#        graph_results(data_filtered[['preds', target_col]])
        graph_

def graph_results(data):
    fig, ax = plt.subplots(figsize=(15,8)) 
    #myFmt = mdates.DateFormatter("%y-%m")
    #ax.xaxis.set_major_formatter(myFmt)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax.set_title('Preds', size= 30)
#    wide_df = data[['preds', target_col]]
    ax = lineplot(data=data)
    plt.xlabel('Year', size=20)
    plt.ylabel(target_col, size=20)
    plt.xticks(rotation=45)
    plt.grid(which='major');


#test all months
for i in range(24):
    print('month {}'.format(i+1))
    bad_model_test(data, 
                   '2016-01-01', 
                   ['TB3SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM'], 
                   'USREC_shift_'+str(i+1)
                   )
    

bad_model_test(data, 
               '2020-01-01', 
               ['TB3SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM'], 
               'future_rec',
               graph=True
               )