#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:07:31 2022

@author: chenwynn
"""
import pandas as pd
import numpy as np
import sys
path = '/Users/chenwynn/Documents/Intern_project/HTSC_Summer/sabr_vol'
sys.path.append(path)
import os
import datetime
import swifter
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
import copy
import pickle
from sabr_bywynn import BlackScholes, SABR

data_path = '/Users/chenwynn/Documents/Intern_project/HTSC_Summer/team/SabrSummerProject2022/03_data'
f = open(os.path.join(data_path, '300etf_2207_sabr_v1.pkl'),'rb')
sabr2207 = pickle.load(f)
sabr2207 = sabr2207.loc[:26173, :]
# f = open(os.path.join(data_path, '300etf_2208_sabr_v1.pkl'),'rb')
# sabr2207 = pickle.load(f)

'''
f = open(os.path.join(data_path, 'greeks_2207_v1.pkl'),'rb')
sabr2207s = pickle.load(f)
sabr2207s = sabr2207s.rename(columns = {'bs_iv':'black vol', 'sabr_iv':'sabr vol', 'strike':'strike price'})
sabr2207s = sabr2207s.reset_index()
'''

f = open(os.path.join(data_path, '300etf.pkl'),'rb')
etf = pickle.load(f)
etf = etf.set_index('time').reindex(pd.Index(np.unique(sabr2207['time']))).reset_index()
etf = etf.rename(columns = {'index':'time'})

'''
volatility spread to generate expected price spread
'''
sabr2207 = sabr2207.loc[(np.abs(sabr2207['delta']) < 0.9)&(np.abs(sabr2207['delta']) > 0.1), :]

def gen_price(x):
    isCall = x['type'].apply(lambda x: True if x == 'C' else False)
    bs = BlackScholes(x['strike price'], x['time_to_mature'].values[0], x['spot price'].values[0], x['riskfree rate'].values[0], isCall)
    x['price'] = bs.black(x['sabr vol'])   
    return x

sabr2207 = sabr2207.groupby('time').apply(gen_price)

#time0 = np.unique(sabr2207['time'])[0]
#time1 = np.unique(sabr2207['time'])[480]
#try1 = sabr2207.loc[(sabr2207['time']>=time0)&(sabr2207['time']<time1), :]
sabr2207['signal'] = sabr2207['close'] - sabr2207['price']
sabr2207[['signal','delta', 'vega', 'gamma', 'theta']] = sabr2207.groupby('code')[['signal','delta', 'vega', 'gamma', 'theta']].shift()

import hedge_update
'''
param_dict_ = {'by_rank':2, 'symmetry':True}
# param_dict_ = {'bar':1e-3}
strat = sabr2207.groupby('time').apply(hedge_update.Option_Split_Rank, **param_dict_)
pcpl1 = hedge_update.Hedge_Transform(strat, etf)
hedge_update.Global_Exist(pcpl1)
pcpl1 = pcpl1.groupby('time').progress_apply(hedge_update.Option_Position_Add).reset_index(drop = True)
# temp.to_csv('/Users/chenwynn/Documents/Intern_project/HTSC_Summer/sabr_vol/add.csv')
hedge_update.Global_Exist(pcpl1)
hedge_dict_ = {'bar':8e-4, 'delta_tolerance':5, 'vega_tolerance':5}
pcpl1 = pcpl1.groupby('time').progress_apply(hedge_update.Hedge_Vega, **hedge_dict_).reset_index(drop = True)

pcpl1.to_pickle('/Users/chenwynn/Documents/Intern_project/HTSC_Summer/sabr_vol/strat.pkl')
'''
param_dict_ = {'profit':1e-3, 'hedge':8e-4, 'delta_tolerance':20, 'vega_tolerance':90, 'trade_volume':100, 'quota':0.5}
pcpl1 = hedge_update.Hedge_Transform(sabr2207, etf)
pcpl1['profit_position'] = 0
hedge_update.Global_Exist(pcpl1)
pcpl1 = pcpl1.groupby('time').progress_apply(hedge_update.Hedge_Open, **param_dict_).reset_index(drop = True)


pcpl1['position'] = pcpl1['profit_position'] + pcpl1['hedge_position']

# 手续费
time0 = pcpl1['time'].sort_values().iloc[0]
timen = pcpl1['time'].sort_values().iloc[-1]

# 开仓价为上一个tick的close
pcpl1['open price'] = pcpl1.groupby('code')['close'].shift()

pcpl1['profit_position_diff'] = pcpl1.groupby('code')['profit_position'].diff()
pcpl1.loc[pcpl1['time'] == time0, 'profit_position_diff'] = pcpl1.loc[pcpl1['time'] == time0, 'profit_position']
# pcpl1.loc[pcpl1['time'] == timen, 'cash_diff'] = pcpl1.loc[pcpl1['time'] == timen, 'position'] * pcpl1.loc[pcpl1['time'] == timen, 'close']
pcpl1['profit_cost'] = np.abs(pcpl1['profit_position_diff']) * 1.3
pcpl1.loc[(pcpl1['position'] < 0)&(pcpl1['profit_position_diff'] < 0), 'profit_cost'] = 0

pcpl1['hedge_position_diff'] = pcpl1.groupby('code')['hedge_position'].diff()
pcpl1.loc[pcpl1['time'] == time0, 'hedge_position_diff'] = pcpl1.loc[pcpl1['time'] == time0, 'hedge_position']
pcpl1.loc[pcpl1['type'] != 'S', 'hedge_cost'] = np.abs(pcpl1.loc[pcpl1['type'] != 'S', 'hedge_position_diff']) * 1.3
pcpl1.loc[(pcpl1['type'] != 'S')&(pcpl1['position'] < 0)&(pcpl1['hedge_position_diff'] < 0), 'hedge_cost'] = 0
pcpl1.loc[pcpl1['type'] == 'S', 'hedge_cost'] = np.abs(pcpl1.loc[pcpl1['type'] == 'S', 'hedge_position_diff']) * 0.4 * pcpl1.loc[pcpl1['type'] == 'S', 'close']
pcpl1.loc[(pcpl1['type'] == 'S')&(pcpl1['hedge_position'] < 0)&(pcpl1['hedge_position_diff'] < 0), 'hedge_cost'] = np.abs(pcpl1.loc[(pcpl1['type'] == 'S')&(pcpl1['hedge_position'] < 0)&(pcpl1['hedge_position_diff'] < 0), 'hedge_position_diff']) * 3 * pcpl1.loc[(pcpl1['type'] == 'S')&(pcpl1['hedge_position'] < 0)&(pcpl1['hedge_position_diff'] < 0), 'open price']

pcpl1['position_diff'] = pcpl1.groupby('code')['position'].diff()
pcpl1.loc[pcpl1['time'] == time0, 'position_diff'] = pcpl1.loc[pcpl1['time'] == time0, 'position']
pcpl1.loc[pcpl1['type'] != 'S', 'cost'] = np.abs(pcpl1.loc[pcpl1['type'] != 'S', 'position_diff']) * 1.3
pcpl1.loc[(pcpl1['type'] != 'S')&(pcpl1['position'] < 0)&(pcpl1['position_diff'] < 0), 'cost'] = 0
pcpl1.loc[pcpl1['type'] == 'S', 'cost'] = np.abs(pcpl1.loc[pcpl1['type'] == 'S', 'position_diff']) * 0.4 * pcpl1.loc[pcpl1['type'] == 'S', 'close']
pcpl1.loc[(pcpl1['type'] == 'S')&(pcpl1['position'] < 0)&(pcpl1['position_diff'] < 0), 'cost'] = np.abs(pcpl1.loc[(pcpl1['type'] == 'S')&(pcpl1['position'] < 0)&(pcpl1['position_diff'] < 0), 'position_diff']) * 3 * pcpl1.loc[(pcpl1['type'] == 'S')&(pcpl1['position'] < 0)&(pcpl1['position_diff'] < 0), 'open price']


# cash = pcpl1.groupby('time')['cash_diff'].sum()

pcpl1['cash'] = pcpl1['close'] - pcpl1['open price']
pcpl1['cash'] = pcpl1['cash'] * pcpl1['position'] * 10000
pcpl1.loc[pcpl1['type'] == 'C', 'out moneyness'] = (-pcpl1.loc[pcpl1['type'] == 'C', 'close'] + pcpl1.loc[pcpl1['type'] == 'C', 'strike price']).apply(lambda x: max(x, 0))
pcpl1.loc[pcpl1['type'] == 'P', 'out moneyness'] = (pcpl1.loc[pcpl1['type'] == 'P', 'close'] - pcpl1.loc[pcpl1['type'] == 'P', 'strike price']).apply(lambda x: max(x, 0))
pcpl1['margin'] = pcpl1.apply(lambda x:max(-(min(x['open price'] + max(0.12*x['spot price'] - x['out moneyness'], 0.07 * x['strike price']), x['strike price'])) * 1e4 * x['position'], x['open price'] * x['position'] * 1e4), axis = 1)
pcpl1['fund_cost'] = pcpl1['margin'] * 0.04/250/48
pcpl1['cost'] = pcpl1['cost'].fillna(0)
pcpl1['fund_cost'] = pcpl1['fund_cost'].fillna(0)
pcpl1['cash_net'] = pcpl1['cash'] - pcpl1['cost'] - pcpl1['fund_cost']
# cash = pcpl1.groupby('time')['cash'].sum()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.title('Return Decomposition')
pcpl1['delta_p'] = pcpl1['delta']*pcpl1['position']
delta_p = pcpl1.groupby('time')['delta_p'].sum()
delta_p = (pcpl1.loc[pcpl1['type'] == 'S', 'close'].diff()).values * delta_p * 1e4
pcpl1['vega_p'] = pcpl1.groupby('code')['sabr vol'].diff()
pcpl1['arbi'] = pcpl1.groupby('code')['black vol'].diff() - pcpl1['vega_p']
pcpl1['arbi'] = pcpl1['arbi'] * pcpl1['vega'].values * pcpl1['position'].values * 1e4
pcpl1['vega_p'] = pcpl1['vega_p'] * pcpl1['vega'].values * pcpl1['position'].values * 1e4
pcpl1['gamma_p'] = pcpl1['gamma']*pcpl1['position'].values
gamma_p = pcpl1.groupby('time')['gamma_p'].sum()
gamma_p = ((pcpl1.loc[pcpl1['type'] == 'S', 'close'].diff()).values)**2 * gamma_p/2 * 1e4
pcpl1['theta_p'] = pcpl1.groupby('code')['time_to_mature'].diff()
pcpl1['theta_p'] = pcpl1['theta_p'] * pcpl1['theta'].values * pcpl1['position'].values * 1e4
plt.plot(delta_p.cumsum(), label='delta')
plt.plot(pcpl1.groupby('time')['vega_p'].sum().cumsum(), label='vega')
plt.plot(gamma_p.cumsum(), label='gamma')
plt.plot(pcpl1.groupby('time')['theta_p'].sum().cumsum(), label='theta')
plt.plot(pcpl1.groupby('time')['arbi'].sum().cumsum(), label = 'vol shift')
plt.legend()


plt.figure(figsize = (10, 5))
plt.title('Strategy Return')
plt.plot(pcpl1.groupby('time')['cash'].sum().cumsum(), label = 'Return Without Cost')
plt.plot(pcpl1.groupby('time')['cash_net'].sum().cumsum(), label = 'Return With Cost')
plt.plot(delta_p.cumsum()+pcpl1.groupby('time')['vega_p'].sum().cumsum()+gamma_p.cumsum()+pcpl1.groupby('time')['theta_p'].sum().cumsum()+pcpl1.groupby('time')['arbi'].sum().cumsum(), label='Greeks Return')
plt.legend()


plt.figure(figsize = (10, 5))
plt.title('Cost Decomposition')
plt.stackplot(np.unique(pcpl1['time']), pcpl1.groupby('time')['profit_cost'].sum().cumsum(), pcpl1.groupby('time')['hedge_cost'].sum().cumsum())
plt.plot(pcpl1.groupby('time')['cost'].sum().cumsum(), color='r')


pcpl1['notional'] = np.abs(pcpl1['position'] * pcpl1['spot price'])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(pcpl1.groupby('time')['notional'].sum(), c='b',ls='--',label='notional amount')
ax.set_xlabel('time', fontsize=16)
ax.set_ylabel('Notional Amount', color='b', fontsize=16)
ax.tick_params('y', colors='b')
ax.legend()

ax2 = ax.twinx()
ax2.plot(pcpl1.loc[pcpl1['position'] != 0, :].groupby('time').size(), c='r', ls=":",label='Position')
ax2.set_ylabel('Contract Number',color='r', fontsize=16)
ax2.tick_params('y', colors='r')
ax2.legend()

plt.title('Position Fluctuate', fontsize=16)
plt.show()   

