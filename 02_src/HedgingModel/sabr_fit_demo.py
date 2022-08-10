#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:33:48 2022

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
import pickle

data_path = '/Users/chenwynn/Documents/Intern_project/HTSC_Summer/team/SabrSummerProject2022/03_data'
f = open(os.path.join(data_path, '300etf_option2207.pkl'),'rb')
option2207 = pickle.load(f)
f = open(os.path.join(data_path, '300etf.pkl'),'rb')
etf = pickle.load(f)

'''
clean data
'''
etf = etf.rename(columns = {'datetime':'time'})
etf['time'] = pd.to_datetime(etf['time'])
option2207 = option2207.rename(columns = {'datetime':'time'})
option2207['time'] = pd.to_datetime(option2207['time'])
option2207['strike price'] = option2207['option_code'].apply(lambda x: int(x[-4:]))/1000
option2207['type'] = option2207['type'].apply(lambda x: 'C' if x == 1 else 'P')
option2207['time_to_mature'] = option2207['time'].apply(lambda x: x.date())
trading_date = np.sort(np.unique(option2207['time_to_mature']))
trading_interval = np.unique(option2207['time'].apply(lambda x: str(x.hour).zfill(2)+str(x.minute).zfill(2)))
option2207['time_to_mature'] = option2207['time_to_mature'].apply(lambda x: len(trading_date) - 1 - np.where(trading_date == x)[0][0])
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
date1 = np.unique(option2207['date'])[-6]
insample = option2207.loc[(option2207['date'] > date0)&(option2207['date'] <= date1), :]

params = {}
param = np.array([0.2, 1, 2.8, -0.2])
r = 0
def pandas_apply_sabr(temp):
    global params
    global r
    global param
    
    # drop deep itm call
    deepcall = temp.loc[(temp['type'] == 'C')&(temp['strike price'] < temp['spot price']-0.3), :]
    dropindex = deepcall.loc[deepcall['close']+deepcall['strike price'] > deepcall['spot price'], :].index
    temp = temp.drop(dropindex)
    # calculate black IV
    y = temp['strike price'].values
    expiry = temp['time_to_mature'].values[0]
    spot_price = temp['spot price'].values[0]
    isCall = temp['type'].apply(lambda x: True if x == 'C' else False).values
    price = temp['close'].values
    atm_spread = y[np.abs(y - spot_price).argmin()] - spot_price
    atm_index_cp = np.where(y == spot_price + atm_spread)[0]
    atm_index_c = atm_index_cp[np.where(isCall[atm_index_cp] == True)[0][0]]
    atm_index_p = atm_index_cp[np.where(isCall[atm_index_cp] == False)[0][0]]
    # use atm put call parity to calculate riskfree interest rate
    r_plus = -np.log((price[atm_index_p] - price[atm_index_c] + spot_price)/y[atm_index_c]) / expiry
    r = (4*r + r_plus)/5
    
    BS = BlackScholes(y, expiry, spot_price, r, isCall)
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
    atm_index_cp = np.where(y == spot_price + atm_spread)[0]
    if len(atm_index_cp) < 2:
        atm_index = atm_index_cp[0]
    else:
        atm_index_c = atm_index_cp[np.where(isCall[atm_index_cp] == True)[0][0]]
        atm_index_p = atm_index_cp[np.where(isCall[atm_index_cp] == False)[0][0]]
        if atm_spread < 0:
            atm_index = atm_index_p
        else:
            atm_index = atm_index_c

    sabr = SABR(y, expiry, spot_price, r, isCall)
    # if want to use different optimization methods, change function here
    param_n = sabr.haganFit_split(sabr.BS, price, atm_index, param[1:])
    print(param_n)
    if param_n[2] > 1.5:
        param = param_n
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

'''
plot to see volatility surface
'''
params1 = copy.deepcopy(params)
params1 = pd.DataFrame(params1)

import matplotlib.pyplot as plt
import matplotlib as mpl
cmap = mpl.cm.get_cmap('RdBu')
plt.figure()
for i in range(48):
    y = np.linspace(4, 4.9, 100)
    df = try1.loc[try1['time'] == params1.columns[i],:]
    expiry = df['time_to_mature'].values[0]
    isCall = df['type'].apply(lambda x: True if x == 'C' else False)
    spot_price = df['spot price'].values[0]
    r = df['riskfree rate'].values[0]
    sabr = SABR(y, expiry, spot_price, r, isCall)
    vol = sabr.haganLogNormalApprox(sabr.BS, params1.iloc[0, i], params1.iloc[1, i], params1.iloc[2, i], params1.iloc[3, i])
    plt.plot(y, vol, c = cmap(i/48))

for i in range(0, 48):
    time = params1.columns[i]
    df = try1.loc[try1['time'] == time,:]
    price = df['close'].values
    y = df['strike price'].values
    expiry = df['time_to_mature'].values[0]
    isCall = df['type'].apply(lambda x: True if x == 'C' else False)
    spot_price = df['spot price'].values[0]
    r = df['riskfree rate'].values[0]
    sabr = SABR(y, expiry, spot_price, r, isCall)
    sabr.haganPlot(sabr.BS, price, params1[time].values)