# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:01:10 2020

@author: rkako
"""

#%% Import Packeges
import os
import vis_utils as vis
import utils as ut

#%% Set Directories and Pathes"""
current_dir = os.getcwd()
working_dir = current_dir #os.path.join(current_dir, 'gdrive/My Drive/MyCOVID19/forecasting/')
dataset_dir = os.path.join(working_dir, 'COVID-19/csse_covid_19_data/csse_covid_19_time_series/')

print('current_dir: ', current_dir)
print('working_dir: ', working_dir)
print('dataset_dir: ', dataset_dir)

#%% Load Data
global_confirmed_df, global_recovered_df, global_deceased_df = ut.get_data(dataset_dir)

#%% Some data stats
num_total_lands, num_total_days = global_confirmed_df.shape
print('num_total_lands: {}, num_total_days: {} '.format(num_total_lands, num_total_days))

#%% Get some of mostly affected lands
num_tops = 20
top_indexes, sum_tops_df, tops_confirmed_df, tops_recovered_df, tops_deceased_df = ut.get_tops(global_confirmed_df, global_recovered_df, global_deceased_df, num_tops)

#%% Recognize the start dates
start_dates, durations = ut.find_start_dates(tops_confirmed_df)

#%% Calculate growth df
delta_t = 1
tops_confirmed_growth_df = ut.get_growth(df=tops_confirmed_df, delta_t=delta_t)
tops_recovered_growth_df = ut.get_growth(df=tops_recovered_df, delta_t=delta_t)
tops_deceased_growth_df = ut.get_growth(df=tops_deceased_df, delta_t=delta_t)

#%% Country-wife processing
land = 'Iran'
land_since_df = ut.get_land_since(tops_confirmed_df, tops_recovered_df, tops_deceased_df, 
                               start_dates, durations, delta_t, land=land)

# for one country
rates_df = ut.get_recovery_mortality_rates(land_since_df)

# for TopQ countries
# mortality_rate, recovery_rate = get_mortality_recovery_rates()

#%% Predictive modeling
from predictive_modeling import XgbReg

x_train, y_train, x_valid, y_valid, x_test, y_test = ut.data_splitor(land_since_df, title='Confirmed')
model = XgbReg()
best = model.train(x_train, y_train, x_valid, y_valid)
y_pred, MSE = model.test(best, x_test, y_test)
print("Root Mean Square Value:", MSE)

Visualization = False
if Visualization:
    vis.bar_plot(top_indexes, sum_tops_df)
    
    vis.timeseries_plot(global_confirmed_df, 'Confirmed Cases')
    vis.timeseries_plot(global_recovered_df, 'Recovered Cases')
    vis.timeseries_plot(global_deceased_df, 'Deceased Cases')
    
    vis.timeseries_plot_since(tops_confirmed_df, 'Confirmed', start_dates, durations)
    vis.timeseries_plot_since(tops_recovered_df, 'Recovered', start_dates, durations)
    vis.timeseries_plot_since(tops_deceased_df, 'Deceased', start_dates, durations)
    vis.growth_plot(tops_confirmed_growth_df, 'Confirmed', delta_t)
    vis.growth_plot(tops_recovered_growth_df, 'Recovered', delta_t)
    vis.growth_plot(tops_deceased_growth_df, 'Deseased', delta_t)
    
    vis.land_plot(land_since_df, land=land)
    vis.rate_plot(rates_df)
    
    # vis.rate_plot(mortality_rate, recovery_rate, land='TopQ')



