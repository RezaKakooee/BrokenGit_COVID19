# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 02:11:04 2020

@author: rkako
"""

#%%
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xgboost as xgb
from sklearn.metrics import mean_squared_error

#%%
class XgbReg:
    def __init__(self):
        self.space = {'n_estimators':     hp.choice('n_estimators', np.arange(100, 500)),
                      'max_depth':        hp.choice('max_depth', np.arange(3, 18, 1, dtype=int)),
                      'gamma':            hp.quniform('gamma', 1, 9, 1),
                      # 'learning_rate':    hp.quniform('learning_rate', 0.01, 1,0.01),
                      'reg_alpha':        hp.quniform('reg_alpha', 40,180,1),
                      'reg_lambda':       hp.quniform('reg_lambda', 0.1, 1, 0.01),
                      'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.5, 1, 0.1)),
                      'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
                      'scale_pos_weight': hp.quniform('scale_pos_weight', 50, 200, 10),
                      'subsample':        hp.quniform('subsample', 0.05, 1, 0.05),
                      'seed':             1300}
        
    def hyperparameter_tuning(self, space):
        reg=xgb.XGBRegressor(objective        = 'reg:squarederror',
                             n_estimators     = 100, #space['n_estimators'], 
                             max_depth        = space['max_depth'], 
                             gamma            = space['gamma'],
                             # learning_rate    = space['learning_rate'],
                             reg_alpha        = space['reg_alpha'],
                             reg_lambda       = space['reg_lambda'],
                             colsample_bytree = space['colsample_bytree'],
                             min_child_weight = space['min_child_weight'],
                             scale_pos_weight = space['scale_pos_weight'],
                             subsample        = space['subsample'],
                             seed             = space['seed'])
        
        evaluation = [(self.x_train, self.y_train), (self.x_valid, self.y_valid)]
        
        reg.fit(self.x_train, self.y_train,
                eval_set=evaluation, eval_metric="rmse",
                early_stopping_rounds=10,verbose=False)
    
        pred = reg.predict(self.x_valid)
        mse= mean_squared_error(self.y_valid, pred)
        return {'loss':mse, 'status': STATUS_OK }
        
    def train(self, x_train, y_train, x_valid, y_valid,):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        
        trials = Trials()
        self.best = fmin(fn=self.hyperparameter_tuning,
                         space=self.space,
                         algo=tpe.suggest,
                         max_evals=100,
                         trials=trials)
        
        return self.best
        
    def test(self, best, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test
        xg_reg = xgb.XGBRegressor(objective        ='reg:squarederror', 
                                  n_estimators     = 10000, 
                                  max_depth        = best['max_depth'],
                                  gamma            = best['gamma'],
                                  # learning_rate    = best['learning_rate'],
                                  reg_alpha        = best['reg_alpha'],
                                  reg_lambda       = best['reg_lambda'],
                                  colsample_bytree = best['colsample_bytree'],
                                  min_child_weight = best['min_child_weight'],
                                  scale_pos_weight = best['scale_pos_weight'],
                                  subsample        = best['subsample'],
                                  seed             = 1300)
        
        xg_reg.fit(self.x_test, self.y_test)
        y_pred = xg_reg.predict(self.x_test)
        
        MSE = np.sqrt(mean_squared_error(self.y_test, y_pred))
        return y_pred, MSE