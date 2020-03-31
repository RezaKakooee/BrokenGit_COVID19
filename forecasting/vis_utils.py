# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import numpy as np
import pandas as pd
from plotly.offline import plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#%%
def bar_plot(top_indexes, sum_tops_df):
    fig = go.Figure(data=[go.Bar(name='Confirmed', x=sum_tops_df['Confirmed'], y=top_indexes, orientation='h', marker_color='royalblue'),
                          go.Bar(name='Recovered', x=sum_tops_df['Recovered'], y=top_indexes, orientation='h', marker_color='Green'),
                          go.Bar(name='Deceased', x=sum_tops_df['Deceased'], y=top_indexes, orientation='h', marker_color='firebrick')])
    
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.show()
    plot(fig)

#%%
def timeseries_plot(df, title, start_date='1/22/20'):
  # df = tops_confirmed_df
  # title = 'Confirmed'
  lands = df.index
  num_lands = len(lands)
  fig = go.Figure()
  for i in range(num_lands):
    xdata = df.columns
    df_land = df.iloc[i,:]
    ydata = df_land[start_date:]
    fig.add_trace(go.Scatter(
                    x=xdata,
                    y=ydata,
                    name=df.index[i],
                    opacity=0.8))
  fig.update_layout(
      title={
          'text': title,
          'y':0.9,
          'x':0.5,
          'xanchor': 'center',
          'yanchor': 'top'},
           xaxis_title="Date",
           yaxis_title="The number of {} cases".format(title))
  fig.show()
  plot(fig)

#%%
def timeseries_plot_since(df, title, start_dates, durations):
  # df = tops_confirmed_df
  # title = 'Confirmed'
  lands = df.index
  fig = go.Figure()
  for ind in df.index:
    xdata = np.arange(0, durations[ind])#df_cince.columns
    df_cince = df.loc[:, start_dates[ind]:]
    ydata = df_cince.loc[ind, start_dates[ind]:].values
    fig.add_trace(go.Scatter(
                    x=xdata,
                    y=ydata,
                    name=ind,
                    opacity=0.8))
  fig.update_layout(
      title={
          'text': title + ' cases',
          'y':0.9,
          'x':0.5,
          'xanchor': 'center',
          'yanchor': 'top'},
      xaxis_title="Number days passed since the first case tested positive",
      yaxis_title="The number of {} cases".format(title))
      
  fig.show()
  plot(fig)

#%%
def growth_plot(df, title, delta_t):
  # df = tops_confirmed_speed
  # title = 'Confirmed'
  lands = df.index
  num_lands = len(lands)
  fig = go.Figure()
  for i in range(num_lands):
    df_land = df.iloc[i,:]
    fig.add_trace(go.Scatter(
                    x=df.columns,
                    y=df_land.values,
                    name=df.index[i],
                    opacity=0.8))
  fig.update_layout(
      title={
          'text': "Growth of {} cases".format(title),
          'y':0.9,
          'x':0.5,
          'xanchor': 'center',
          'yanchor': 'top'},
          xaxis_title="Every {} days".format(delta_t),
          yaxis_title="The number of {} cases".format(title))
  fig.show()
  plot(fig)

#%%
def growth_plot_since(df, title, delta_t):
  # df = tops_confirmed_speed
  # title = 'Confirmed'
  lands = df.index
  num_lands = len(lands)
  fig = go.Figure()
  for i in range(num_lands):
    df_land = df.iloc[i,:]
    fig.add_trace(go.Scatter(
                    x=df.columns,
                    y=df_land.values,
                    name=df.index[i],
                    opacity=0.8))
  fig.update_layout(
      title={
          'text': "Growth of {} cases".format(title),
          'y':0.9,
          'x':0.5,
          'xanchor': 'center',
          'yanchor': 'top'},
          xaxis_title="Every {} days".format(delta_t),
          yaxis_title="The number of {} cases".format(title))
  fig.show()
  plot(fig)
  
#%%
def land_plot(land_since_df, land='Iran'):
  
  fig = make_subplots(rows=2, cols=1)
  xdata = land_since_df.index
  fig.add_trace(go.Scatter(x=xdata, y=land_since_df['Confirmed'].values,
                           name='Confirmed', opacity=0.8, mode='lines', line = dict(color='royalblue')), row=1, col=1)
  
  fig.add_trace(go.Scatter(x=xdata, y=land_since_df['Recovered'],
                           name='Recovered', opacity=0.8, mode='lines', line = dict(color='firebrick')), row=1, col=1)

  fig.add_trace(go.Scatter(x=xdata, y=land_since_df['Deceased'],
                           name='Deceased', opacity=0.8, mode='lines', line = dict(color='green')), row=1, col=1)

  fig.add_trace(go.Scatter(x=xdata, y=land_since_df['Confirmed Growth'].values,
                           name='Confirmed Speed', opacity=0.8, mode='lines', line = dict(color='royalblue')), row=2, col=1)
  
  fig.add_trace(go.Scatter(x=xdata, y=land_since_df['Recovered Growth'].values,
                           name='Recovered Growth', opacity=0.8, mode='lines', line = dict(color='firebrick')), row=2, col=1)
  
  fig.add_trace(go.Scatter(x=xdata, y=land_since_df['Deceased Growth'].values,
                           name='Deceased Growth', opacity=0.8, mode='lines', line = dict(color='green')), row=2, col=1)
  

  #### Mean
  mean_confirmed = land_since_df.describe().mean()[0]
  mean_recovered = land_since_df.describe().mean()[1]
  mean_deceased = land_since_df.describe().mean()[2]
  mean_confirmed_growth = land_since_df.describe().mean()[3]
  mean_recovered_growth = land_since_df.describe().mean()[4]
  mean_deceased_growth = land_since_df.describe().mean()[5]
  
  num_days = len(xdata)
  confirmed_mean_vec = mean_confirmed*np.ones(num_days)
  deceased_mean_vec = mean_recovered*np.ones(num_days)
  recovered_mean_vec = mean_deceased*np.ones(num_days)
  confirmed_speed_mean_vec = mean_confirmed_growth*np.ones(num_days)
  deceased_speed_mean_vec = mean_recovered_growth*np.ones(num_days)
  recovered_speed_mean_vec = mean_deceased_growth*np.ones(num_days)

  fig.add_trace(go.Scatter(x=xdata, y=confirmed_mean_vec,
                           name='Confirmed Mean', opacity=0.8, line = dict(color='royalblue', dash='dash')), row=1, col=1)
  
  fig.add_trace(go.Scatter(x=xdata, y=deceased_mean_vec,
                           name='Deceased Mean', opacity=0.8, line = dict(color='firebrick', dash='dash')), row=1, col=1)

  fig.add_trace(go.Scatter(x=xdata, y=recovered_mean_vec,
                           name='Recovered Meam', opacity=0.8, line = dict(color='green', dash='dash')), row=1, col=1)
  
  fig.add_trace(go.Scatter(x=xdata, y=confirmed_speed_mean_vec,
                           name='Confirmed Growth Mean', opacity=0.8, line = dict(color='royalblue', dash='dash')), row=2, col=1)
  
  fig.add_trace(go.Scatter(x=xdata, y=deceased_speed_mean_vec,
                           name='Deceased Growth Mean', opacity=0.8, line = dict(color='firebrick', dash='dash')), row=2, col=1)

  fig.add_trace(go.Scatter(x=xdata, y=recovered_speed_mean_vec,
                           name='Recovered Growth Meam', opacity=0.8, line = dict(color='green', dash='dash')), row=2, col=1)
  
  fig.update_layout(title={'text': land, 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})
  fig.show()
  plot(fig)

#%%
def rate_plot(rates_df, land='Iran'):
  fig = go.Figure()
  xdata = rates_df.index
  fig.add_trace(go.Scatter(x=xdata, y=rates_df['Recovery'],
                           name='Recovery Rate', opacity=0.8, mode='lines', 
                           line = dict(color='green')))

  fig.add_trace(go.Scatter(x=xdata, y=rates_df['Mortality'],
                           name='Mortality Rate', opacity=0.8, mode='lines', 
                           line = dict(color='firebrick')))
  fig.update_layout(title={'text': 'Mortality and Recovery Rates in {}'.format(land), 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})
  fig.show()
  plot(fig)