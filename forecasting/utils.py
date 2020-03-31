# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 03:34:01 2020

@author: rkako
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pandas_profiling import ProfileReport

#%%
def get_data(dataset_dir):
    confirmed_path = os.path.join(dataset_dir, 'time_series_covid19_confirmed_global.csv')
    deceased_path = os.path.join(dataset_dir, 'time_series_covid19_deaths_global.csv')
    recovered_path = os.path.join(dataset_dir, 'time_series_covid19_recovered_global.csv')  
    
    confirmed_df = pd.read_csv(confirmed_path, index_col=False)
    recovered_df = pd.read_csv(recovered_path, index_col=False)
    deceased_df = pd.read_csv(deceased_path, index_col=False)
    
    confirmed_df.drop(columns=['Lat', 'Long', 'Province/State'], inplace=True)
    recovered_df.drop(columns=['Lat', 'Long', 'Province/State'], inplace=True)
    deceased_df.drop(columns=['Lat', 'Long', 'Province/State'], inplace=True)
    
    
    confirmed_df = confirmed_df.groupby(['Country/Region']).sum()
    recovered_df = recovered_df.groupby(['Country/Region']).sum()
    deceased_df = deceased_df.groupby(['Country/Region']).sum()
    
    return confirmed_df, recovered_df, deceased_df
    
#%%
def get_tops(confirmed_df, recovered_df, deceased_df, num_tops):

    sum_confirmed_df = confirmed_df.iloc[:,-1]
    sum_recovered_df = recovered_df.iloc[:,-1]
    sum_deceased_df = deceased_df.iloc[:,-1]
    
    copy_sum_confirmed_df = sum_confirmed_df.copy()
    copy_sum_confirmed_df.sort_values(ascending=False, inplace=True)
    sum_tops_confirmed = copy_sum_confirmed_df.iloc[0:num_tops]
       
    sum_others_confirmed = copy_sum_confirmed_df.iloc[num_tops:]
    
    top_indexes = sum_tops_confirmed.index
    other_indexes = sum_others_confirmed.index
    
    sum_tops_recovered = sum_recovered_df.drop(other_indexes, axis=0)
    
    sum_tops_deceased = sum_deceased_df.drop(other_indexes, axis=0)
    
    sum_tops_df = pd.concat([sum_tops_confirmed, sum_tops_recovered, sum_tops_deceased], axis=1, sort=False)
    sum_tops_df.columns = ['Confirmed', 'Recovered', 'Deceased']
    
    tops_confirmed_df = confirmed_df.loc[top_indexes, :]
    tops_recovered_df = recovered_df.loc[top_indexes, :]
    tops_deceased_df = deceased_df.loc[top_indexes, :]
        
        
    return top_indexes, sum_tops_df, tops_confirmed_df, tops_recovered_df, tops_deceased_df
 
    #%%
def get_pd_profiler(global_confirmed_df, global_recovered_df, global_deceased_df, working_dir):
  profile_confirmed = ProfileReport(global_confirmed_df)
  profile_recovered = ProfileReport(global_recovered_df)
  profile_deceased = ProfileReport(global_deceased_df)
  profile_confirmed.to_file(outputfile=os.path.join(working_dir, "./pd_profiling/Profiling for Global Confirmed Cases.html"))
  profile_recovered.to_file(outputfile=os.path.join(working_dir, "./pd_profiling/Profiling for Global Recovered Cases.html"))
  profile_deceased.to_file(outputfile=os.path.join(working_dir, "./pd_profiling/Profiling for Global Deceased Cases.html"))

#%%
def get_land_since(tops_confirmed_df, tops_recovered_df, 
                   tops_deceased_df, start_dates, durations, delta_t, land='Iran'):
    
  land_confirmed_since_df = tops_confirmed_df.loc[land,:].loc[start_dates[land]:]
  land_recovered_since_df = tops_recovered_df.loc[land,:].loc[start_dates[land]:]
  land_deceased_since_df = tops_deceased_df.loc[land,:].loc[start_dates[land]:]
  
  date_cols = tops_confirmed_df.loc[:, start_dates[land]:].columns
  num_days = durations[land]+1
  growth_list_C = []
  growth_list_R = []
  growth_list_D = []
  for i in range(0, num_days-delta_t, delta_t):
    growth_list_C.append(land_confirmed_since_df[date_cols[i+delta_t]] - land_confirmed_since_df[date_cols[i]])
    growth_list_R.append(land_recovered_since_df[date_cols[i+delta_t]] - land_recovered_since_df[date_cols[i]])
    growth_list_D.append(land_deceased_since_df[date_cols[i+delta_t]] -  land_deceased_since_df[date_cols[i]])
  
  begining_gap_growth = len(growth_list_C)
  list_temp = list(np.zeros(num_days-begining_gap_growth))
  
  growth_list_C = np.insert(np.array(growth_list_C), 0, list_temp)
  growth_list_R = np.insert(np.array(growth_list_R), 0, list_temp)
  growth_list_D = np.insert(np.array(growth_list_D), 0, list_temp)
 
  
  land_since_df = pd.DataFrame({'Confirmed': land_confirmed_since_df.values,
                                'Recovered': land_recovered_since_df.values,
                                'Deceased': land_deceased_since_df.values,
                                'Confirmed Growth': growth_list_C,
                                'Recovered Growth': growth_list_R,
                                'Deceased Growth': growth_list_D,}, 
                                 index = np.arange(num_days))
  
  return land_since_df

#%%
def find_start_dates(confirmed):
  # very_start_date = confirmed.columns[0]
  very_end_date = confirmed.columns[-1]
  lands = confirmed.index
  start_dates = {}
  for land in lands:
    df1 = confirmed.loc[confirmed.index==land].iloc[0,1:]!=0
    start_dates.update({land: df1.index[df1==True].tolist()[0]})

  durations = {}
  for key in start_dates.keys():
      sd = start_dates[key]
      d1 = datetime.strptime(sd, '%m/%d/%y').date()
      d2 = datetime.strptime(very_end_date, '%m/%d/%y').date()
      delta = d2 - d1
      durations.update({key: delta.days})
      
  return start_dates, durations

#%%
def get_growth(df, delta_t=1):
  land_names=df.index
  date_cols = df.columns
  num_days= len(date_cols)

  growth_list = []
  for i in range(0, num_days-delta_t, delta_t):
    growth_list.append(df[date_cols[i+delta_t]].values - df[date_cols[i]].values)
  growth_arr = np.array(growth_list)

  new_date_cols = [date_cols[0]+'-'+date_cols[i] for i in range(1, num_days-delta_t, delta_t)]
  growth_df = pd.DataFrame(columns=new_date_cols, index=land_names)

  for i, ncol in enumerate(new_date_cols):
    growth_df[ncol] = growth_arr[i,:]
  # speed_df.insert(0, 'Land', land_names.values)

  return growth_df

#%%
def get_recovery_mortality_rates(land_since_df):
  confirmed1 = land_since_df['Confirmed']
  recovered1 = land_since_df['Recovered']
  deceased1 = land_since_df['Deceased']
  recovery_rate = recovered1 / confirmed1
  mortality_rate = deceased1 / confirmed1
  
  rates_df = pd.DataFrame({ 'Recovery': recovery_rate, 'Mortality': mortality_rate})
  
  return rates_df