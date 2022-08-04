#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:07:31 2022

@author: chenwynn
"""
import pandas as pd
import numpy as np
import sys
# path = '/Users/chenwynn/Documents/Intern_project/HTSC_Summer/team/SabrSummerProject2022/02_src'
# sys.path.append(path)
import os
import datetime
import swifter
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
import copy

data_path = '/Users/chenwynn/Documents/Intern_project/HTSC_Summer/team/SabrSummerProject2022/03_data'
option2207 = pd.read_csv(os.path.join(data_path, '300etf_option2207.csv'), index_col=0)
etf = pd.read_csv(os.path.join(data_path, '300etf.csv'), index_col=0)

'''
clean data
'''
etf = etf.rename(columns = {'datetime':'time'})
etf['time'] = pd.to_datetime(etf['time'])
option2207 = option2207.rename(columns = {'datetime':'time'})
option2207['time'] = pd.to_datetime(option2207['time'])
option2207['strike price'] = option2207['option_code'].apply(lambda x: int(x[-4:]))/1000
option2207['type'] = option2207['type'].apply(lambda x: 'C' if x == '看涨期权' else 'P')
option2207['time_to_mature'] = option2207['time'].apply(lambda x: x.date())
trading_date = np.sort(np.unique(option2207['time_to_mature']))
trading_interval = np.unique(option2207['time'].apply(lambda x: str(x.hour).zfill(2)+str(x.minute).zfill(2)))
option2207['time_to_mature'] = option2207['time_to_mature'].apply(lambda x: len(trading_date) + 1 - np.where(trading_date == x)[0][0])
option2207['time_to_mature'] = option2207['time_to_mature'] * len(trading_interval) + option2207['time'].apply(lambda x: len(trading_interval) - 1 - np.where(trading_interval == (str(x.hour).zfill(2)+str(x.minute).zfill(2)))[0][0])
trading_year = len(trading_interval) * 250
option2207['time_to_mature'] = option2207['time_to_mature'] / trading_year

'''
option model
'''
sys.path.append('/Users/chenwynn/Documents/Intern_project/HTSC_Summer/sabr_vol')
from sabr import BlackScholes, SABR

option2207['date'] = option2207['time'].apply(lambda x: x.date())
option2207 = pd.merge(option2207, etf.rename(columns = {'close':'spot price'})[['time','spot price']], how = 'left', on  = 'time')
date0 = np.unique(option2207['date'])[-7]
date1 = np.unique(option2207['date'])[-1]
insample = option2207.loc[(option2207['date'] > date0)&(option2207['date'] <= date1), :]

params = {}
r = -0.01
def pandas_apply_sabr(temp):
    global params
    global r
    
    # drop deep itm call
    deepcall = temp.loc[(temp['type'] == 'C')&(temp['strike price'] < temp['spot price']-0.3), :]
    dropindex = deepcall.loc[deepcall['close']+deepcall['strike price'] > deepcall['spot price'], :].index
    temp = temp.drop(dropindex)
    # calculate black IV
    y = temp['strike price'].values
    expiry = temp['time_to_mature'].values[0]
    spot_price = temp['spot price'].values[0]
    isCall = temp['type'].apply(lambda x: True if x == 'C' else False)
    price = temp['close'].values
    BS = BlackScholes(y, expiry, spot_price, 0.04, isCall)
    bvol = BS.blackIV(price)
    # drop vol outlier
    bvol[bvol<0.08] = bvol[bvol>0.08].min()
    bvol[bvol>0.5] = np.NAN
    # check
    if len(bvol[~np.isnan(bvol)]) < 3:
        return temp
    # drop 10 delta
    bdelta = BS.blackDelta(bvol)
    argfit = (np.abs(bdelta) > 0.1)&(np.abs(bdelta) < 0.9)
    # fit sabr
    y = y[argfit]
    isCall = isCall[argfit]
    price = price[argfit]
    # volume = temp['volume'].values[argfit]
    # volume[volume < 1] = 1
    atm_spread = y[np.abs(y - spot_price).argmin()] - spot_price
    atm_index_c = np.where(y[isCall] == spot_price + atm_spread)
    atm_index_c = np.where(price == price[isCall][atm_index_c])
    atm_index_p = np.where(y[~isCall] == spot_price + atm_spread)
    atm_index_p = np.where(price == price[~isCall][atm_index_p])
    if atm_spread < 0:
        atm_index = atm_index_p
    else:
        atm_index = atm_index_c
    # use atm put call parity to calculate riskfree interest rate
    r_plus = -np.log((price[atm_index_p] - price[atm_index_c] + spot_price)/y[atm_index]) / expiry
    r = (4*r+r_plus[0])/5
    sabr = SABR(y, expiry, spot_price, r, isCall)
    param= sabr.haganFit_delta(sabr.BS, price, atm_index)
    temp = temp.loc[argfit, :]
    params[temp['time'].values[0]] = param
    temp['black vol'] = bvol[argfit]
    temp['sabr vol'] = sabr.haganLogNormalApprox(sabr.BS, param[0], param[1], param[2], param[3])
    temp['delta'] = sabr.computeSABRdelta(sabr.BS, param[0], param[1], param[2], param[3])
    temp['vega'] = sabr.computeSABRvega(sabr.BS, param[0], param[1], param[2], param[3])
    temp['theta'] = sabr.computeSABRtheta(sabr.BS, param[0], param[1], param[2], param[3])
    temp['gamma'] = sabr.computeSABRgamma(sabr.BS, param[0], param[1], param[2], param[3])
    temp['riskfree rate'] = r
    
    return temp


try1 = insample.groupby('time').progress_apply(pandas_apply_sabr).reset_index(drop = True)

# plot to see volatility surface
params1 = copy.deepcopy(params)
params1 = pd.DataFrame(params1)

import matplotlib.pyplot as plt
import matplotlib as mpl
cmap = mpl.cm.get_cmap('RdBu')
plt.figure()
for i in range(240):
    y = np.linspace(4, 4.9, 100)
    df = try1.loc[try1['time'] == params1.columns[i],:]
    expiry = df['time_to_mature'].values[0]
    isCall = df['type'].apply(lambda x: True if x == 'C' else False)
    spot_price = df['spot price'].values[0]
    sabr = SABR(y, expiry, spot_price, 0.04, isCall)
    vol = sabr.haganLogNormalApprox(sabr.BS, params1.iloc[0, i], params1.iloc[1, i], params1.iloc[2, i], params1.iloc[3, i])
    plt.plot(y, vol, c = cmap(i/240))

for i in range(0, 12):
    time = params1.columns[i]
    df = try1.loc[try1['time'] == time,:]
    price = df['close'].values
    y = df['strike price'].values
    expiry = df['time_to_mature'].values[0]
    isCall = df['type'].apply(lambda x: True if x == 'C' else False)
    spot_price = df['spot price'].values[0]
    sabr = SABR(y, expiry, spot_price, 0.04, isCall)
    sabr.haganPlot(sabr.BS, price, params1[time].values)

'''
volatility spread to generate expected price spread
'''
try1 = try1.loc[(np.abs(try1['delta']) < 0.9)&(np.abs(try1['delta']) > 0.1), :]

def gen_price(x):
    isCall = x['type'].apply(lambda x: True if x == 'C' else False)
    bs = BlackScholes(x['strike price'], x['time_to_mature'].values[0], x['spot price'].values[0], x['riskfree rate'].values[0], isCall)
    x['price'] = bs.black(x['sabr vol'])   
    return x

try1 = try1.groupby('time').apply(gen_price)

try1['signal'] = try1['close'] - try1['price']

etf_insample = etf.set_index('time').reindex(try1.set_index('time').index.drop_duplicates()).reset_index()


# just a demo
# hedgemodel = path + '/HedgingModel/hedge.py'
# os.system('python '+hedgemodel)
import hedge

dict_ = {'by_rank':1}
pcpl = try1.groupby('time').progress_apply(hedge.Put_Call_Parity_l, **dict_).reset_index(drop = True)
pcps = try1.groupby('time').progress_apply(hedge.Put_Call_Parity_s, **dict_).reset_index(drop = True)
pcpl['profit_position'] = pcpl['profit_position'].fillna(0) + pcps['profit_position'].fillna(0)

pcpl1 = hedge.Hedge_Transform(pcpl, etf_insample)
hedge.Global_Exist(pcpl1)
pcpl1 = pcpl1.groupby('time').progress_apply(hedge.Option_Position_Hold).reset_index(drop = True)
pcpl1['hedge_freq'] = pcpl1['time'].apply(lambda x: x.date())
# close_time = pcpl1.groupby('hedge_freq').last().set_index('time').index
# pcpl1_hedge = pcpl1.set_index('time').loc[close_time,:].reset_index()
hedge.Global_Exist(pcpl1)
# pcpl1_hedge = pcpl1_hedge.groupby('time').progress_apply(hedge.Hedge_Spot).reset_index(drop = True)
# pcpl1 = pd.merge(pcpl1.drop(columns = 'hedge_position'), pcpl1_hedge[['time', 'hedge_position']], how = 'left', on = 'time')
pcpl1 = pcpl1.groupby('time').progress_apply(hedge.Hedge_Spot).reset_index(drop = True)

data_dir = '/Users/chenwynn/Documents/Intern_project/HTSC_Summer/stra'
# pcpl1.to_csv(os.path.join(data_dir, 'strategy_try.csv'))
pcpl1['position'] = pcpl1['profit_position'] + pcpl1['hedge_position']

# 手续费
time0 = pcpl1['time'].sort_values().iloc[0]
timen = pcpl1['time'].sort_values().iloc[-1]
pcpl1['position_diff'] = pcpl1.groupby('code')['position'].diff()
pcpl1.loc[pcpl1['time'] == time0, 'position_diff'] = pcpl1.loc[pcpl1['time'] == time0, 'position']
# pcpl1.loc[pcpl1['time'] == timen, 'cash_diff'] = pcpl1.loc[pcpl1['time'] == timen, 'position'] * pcpl1.loc[pcpl1['time'] == timen, 'close']
pcpl1.loc[pcpl1['type'] != 'S', 'cost'] = np.abs(pcpl1.loc[pcpl1['type'] != 'S', 'position_diff']) * 1.5
pcpl1.loc[(pcpl1['type'] != 'S')&(pcpl1['position'] < 0)&(pcpl1['position_diff'] < 0), 'cost'] = 0
pcpl1.loc[pcpl1['type'] == 'S', 'cost'] = np.abs(pcpl1.loc[pcpl1['type'] == 'S', 'position_diff']) * 0.4 * pcpl1.loc[pcpl1['type'] == 'S', 'close']
pcpl1.loc[(pcpl1['type'] == 'S')&(pcpl1['position'] < 0)&(pcpl1['position_diff'] < 0), 'cost'] = np.abs(pcpl1.loc[(pcpl1['type'] == 'S')&(pcpl1['position'] < 0)&(pcpl1['position_diff'] < 0), 'position_diff']) * 3 * pcpl1.loc[(pcpl1['type'] == 'S')&(pcpl1['position'] < 0)&(pcpl1['position_diff'] < 0), 'close']

# cash = pcpl1.groupby('time')['cash_diff'].sum()

pcpl1['cash'] = pcpl1.groupby('code')['close'].diff()
pcpl1['cash'] = pcpl1.groupby('code')['cash'].shift(-1)
pcpl1['cash'] = pcpl1['cash'] * pcpl1['position'] * 10000
pcpl1['cash'] = pcpl1['cash'] - pcpl1['cost']
cash = pcpl1.groupby('time')['cash'].sum()

import matplotlib.pyplot as plt

plt.figure()
plt.plot(cash.cumsum())






