#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:07:31 2022

@author: chenwynn
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/chenwynn/Documents/Intern_project/HTSC_Summer/team/SabrSummerProject2022/02_src')
import os
import datetime

data_path = '/Users/chenwynn/Documents/Intern_project/HTSC_Summer/team/SabrSummerProject2022/03_data'
option2207 = pd.read_csv(os.path.join(data_path, '300etf_option2207.csv'), index_col=0)
etf = pd.read_csv(os.path.join(data_path, '300etf.csv'), index_col=0)

# data cleaning
etf = etf.rename(columns = {'datetime':'time'})
etf['time'] = pd.to_datetime(etf['time'])
option2207 = option2207.rename(columns = {'datetime':'time'})
option2207['time'] = pd.to_datetime(option2207['time'])
option2207['strike price'] = option2207['option_code'].apply(lambda x: int(x[-4:]))
option2207['type'] = option2207['type'].apply(lambda x: 'C' if x == '看涨期权' else 'P')
option2207['time_to_mature'] = option2207['time'].apply(lambda x: x.date())
trading_date = np.sort(np.unique(option2207['time_to_mature']))
option2207['time_to_maturity'] = option2207['time_to_mature'].apply(lambda x: len(trading_date) - np.where(trading_date == x)[0][0])

# try to build option model, but problems encountered
from OptionModel.sabr import SabrHagan2002
from OptionModel.option_abc import OptABC

def pandas_apply_sabr(df):
    spot = etf.loc[etf['time'] == list(set(df['time']))[0], 'close'].values[0]
    opt_price = df['close'].values
    strike = df['strike price'].values/1000
    texp = df['time_to_maturity'].values[0]/250
    
    option_abc = OptABC(sigma=0.2, intr=0.03)
    sabrmodel = SabrHagan2002(sigma=0.2, vov=0.6, rho=0.3, beta=0.6)
    sabrmodel.calibrate3(price_or_vol3=opt_price, strike3=strike, spot=spot, texp=texp, is_vol=False, setval=True)

# just a demo, if want to rolling through the dataframe, use pd.dataframe.apply(*)
from HedgingModel import hedge
time0 = option2207['time'].values[0]
df = option2207.loc[option2207['time'] == time0, :]
# data is not complete, random delta just for show
df['signal'] = np.random.rand(len(df)) - 0.5
df['delta'] = np.random.rand(len(df)) * 2 - 1

df = hedge.Put_Call_Parity_s(df, 1)
# hedge.Straddle(df)
# hedge.Bull_Call_Spread(df)
# hedge.Option_Split_Rank(df)

df = hedge.hedge_transform(df, etf.loc[etf['time'] == time0, :])
# this step would create a global variable exist_position used to remember current position

df = hedge.Option_Position_Hold(df)

hedge.Hedge_Spot(df)








