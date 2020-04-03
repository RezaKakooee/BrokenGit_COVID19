# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 02:11:04 2020

@author: rkako
"""

#%%
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

import utils as ut
import vis_utils as vis

#%%
tops_confirmed_df = pd.read_csv('tops_confirmed.csv', index_col=False)
tops_recovered_df = pd.read_csv('tops_recovered.csv', index_col=False)
tops_deceased_df = pd.read_csv('tops_deceased.csv', index_col=False)

tops_confirmed_df.set_index('Country/Region', inplace=True)
tops_recovered_df.set_index('Country/Region', inplace=True)
tops_deceased_df.set_index('Country/Region', inplace=True)

top_land_names = tops_confirmed_df.index
start_dates, durations = ut.find_start_dates(tops_confirmed_df)

land = 'Iran'
TITLE = 'Confirmed'
land_since_df = ut.get_land_since(tops_confirmed_df, tops_recovered_df, tops_deceased_df, 
                               start_dates, durations, delta_t=1, land=land)


x_train, y_train, x_valid, y_valid, x_test, y_test = ut.data_splitter(land_since_df, title=TITLE)

space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180}

# Optimization 
def hyperparameter_tuning(space):
    reg=xgb.XGBRegressor(objective ='reg:squarederror',
                         n_estimators=1000,
                         max_depth = int(space['max_depth']), 
                         gamma = space['gamma'],
                         reg_alpha = int(space['reg_alpha']),
                         min_child_weight=space['min_child_weight'],
                         colsample_bytree=space['colsample_bytree'])
    
    evaluation = [(x_train, y_train), (x_valid, y_valid)]
    
    reg.fit(x_train, y_train,
            eval_set=evaluation, eval_metric="rmse",
            early_stopping_rounds=10,verbose=False)

    pred = reg.predict(x_valid)
    mse= mean_squared_error(y_valid, pred)
    # print ("SCORE:", mse)
    return {'loss':mse, 'status': STATUS_OK }

trials = Trials()
best = fmin(fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print (best)

#%% test
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', 
                          colsample_bytree = best['colsample_bytree'], 
                          gamma = best['gamma'],
                          learning_rate = 0.1,
                          min_child_weigh=best['min_child_weight'],
                          max_depth = int(best['max_depth']), 
                          reg_alpha = best['reg_alpha'],
                          reg_lambda = best['reg_lambda'],
                          alpha = 10, 
                          n_estimators = 5000)

xg_reg.fit(x_test, y_test)

y_pred = xg_reg.predict(x_test)

print("Root Mean Square Value:", np.sqrt(mean_squared_error(y_test, y_pred)))


#%% visualize the predictions 
len_y_pred = len(y_pred)
y_real = land_since_df['Confirmed'].values[-len_y_pred:]
vis.plot_predictions(y_real, y_pred, title='Confirmed')
