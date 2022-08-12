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

param_dict_ = {'by_rank':2, 'symmetry':False}
strat = sabr2207.groupby('time').apply(hedge_update.Option_Split_Rank, **param_dict_)
pcpl1 = hedge_update.Hedge_Transform(strat, etf)
hedge_update.Global_Exist(pcpl1)
pcpl1 = pcpl1.groupby('time').progress_apply(hedge_update.Option_Position_Add).reset_index(drop = True)
# temp.to_csv('/Users/chenwynn/Documents/Intern_project/HTSC_Summer/sabr_vol/add.csv')
hedge_update.Global_Exist(pcpl1)
hedge_dict_ = {'bar':8e-4, 'delta_tolerance':50, 'vega_tolerance':100}
pcpl1 = pcpl1.groupby('time').progress_apply(hedge_update.Hedge_Vega, **hedge_dict_).reset_index(drop = True)

# pcpl1.to_pickle('/Users/chenwynn/Documents/Intern_project/HTSC_Summer/sabr_vol/strat.pkl')






