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

data['T1YFFM_1'] = data['T1YFFM'].shift(1)
data['T5YFFM_1'] = data['T5YFFM'].shift(1)
data['T10YFFM_1'] = data['T10YFFM'].shift(1)
data['TB3SMFFM_1'] = data['TB3SMFFM'].shift(1)

data['T1YFFM_2'] = data['T1YFFM'].shift(2)
data['T5YFFM_2'] = data['T5YFFM'].shift(2)
data['T10YFFM_2'] = data['T10YFFM'].shift(2)
data['TB3SMFFM_2'] = data['TB3SMFFM'].shift(2)

#Create all target variables
for i in range(18):
    data['future_rec_6m_{}'.format(str(i))] = data['USREC'].shift(-1*i).iloc[::-1].rolling(6, min_periods=0).sum().iloc[::-1]

data = data.fillna(0)



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
    
    regr = MLPRegressor(hidden_layer_sizes= (10, 10, 10))
    model = regr.fit(scaled_features, scaled_target)
    predictions = model.predict(scaled_features)
    print(mse(predictions, scaled_target))
    if graph:
#        graph_results(data_filtered[['preds', target_col]])
        graph_results(data_filtered.index, 
                      np.ravel(scaler_target.inverse_transform([predictions])), 
                      np.ravel(scaler_target.inverse_transform([scaled_target]))
                      )

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
    plt.grid(which='major');


##test all months
#for i in range(18):
#    print('month {}: '.format(i), end='')
#    bad_model_test(data, 
#                   '2016-01-01', 
#                   ['TB3SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM', 
#                    'TB3SMFFM_1', 'T1YFFM_1', 'T5YFFM_1', 'T10YFFM_1',
#                    'TB3SMFFM_2', 'T1YFFM_2', 'T5YFFM_2', 'T10YFFM_2'], 
#                   'future_rec_6m_{}'.format(str(i))
#                   )
#    

bad_model_test(data, 
    
               '2020-01-01', 
               ['TB3SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM', 
                'TB3SMFFM_1', 'T1YFFM_1', 'T5YFFM_1', 'T10YFFM_1',
                'TB3SMFFM_2', 'T1YFFM_2', 'T5YFFM_2', 'T10YFFM_2'], 
               'future_rec_6m_8',
               graph=True
               )