#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:07:31 2022

@author: chenwynn
"""
import pandas as pd
import numpy as np
import sys
#path = '/Users/chenwynn/Documents/Intern_project/HTSC_Summer/team/SabrSummerProject2022/02_src'
path = '/Users/lenovo/git_intro/SabrSummerProject2022/02_src'
sys.path.append(path)
import os
import datetime
import swifter
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')

#data_path = '/Users/chenwynn/Documents/Intern_project/HTSC_Summer/team/SabrSummerProject2022/03_data'
data_path = '/Users/lenovo/git_intro/SabrSummerProject2022/03_data'
option2207 = pd.read_csv(os.path.join(data_path, '300etf_option2207.csv'), index_col=0)
etf = pd.read_csv(os.path.join(data_path, '300etf.csv'), index_col=0)

# data cleaning
etf = etf.rename(columns = {'datetime':'time'})
etf['time'] = pd.to_datetime(etf['time'])
option2207 = option2207.rename(columns = {'datetime':'time'})
option2207['time'] = pd.to_datetime(option2207['time'])
option2207['strike price'] = option2207['option_code'].apply(lambda x: int(x[-4:]))/1000
option2207['type'] = option2207['type'].apply(lambda x: 'C' if x == '看涨期权' else 'P')
option2207['time_to_mature'] = option2207['time'].apply(lambda x: x.date())
trading_date = np.sort(np.unique(option2207['time_to_mature']))
option2207['time_to_mature'] = option2207['time_to_mature'].apply(lambda x: len(trading_date) - np.where(trading_date == x)[0][0])

# try to build option model, but problems encountered
from OptionModel.sabr import SabrHagan2002
from OptionModel.option_abc import OptABC

def pandas_apply_sabr(df):
    
    def fit_split_cp(df, cp, spot, texp):
        atm = df.loc[np.abs(df['strike price'] - spot).nsmallest(2).index, :]
        index_fit = [atm['volume'].idxmax()]
        index_fit.append(df.loc[df['strike price'] < atm['strike price'].min(), 'volume'].idxmax())
        index_fit.append(df.loc[df['strike price'] > atm['strike price'].max(), 'volume'].idxmax())
        fit = df.loc[index_fit, :].sort_values(by = 'strike price')
        
        opt_price = fit['close'].values
        strike = fit['strike price'].values
        sabrmodel = SabrHagan2002(sigma=0.2, vov=0.6, rho=0.3, beta=0.6)
        sabrmodel.calibrate3(price_or_vol3=opt_price, strike3=strike, spot=spot, texp=texp, is_vol=False, setval=True, cp = cp)
        
        opt_price = df['close'].values
        strike = df['strike price'].values
        df['black Vol'] = sabrmodel.impvol_brentq(opt_price, strike, spot, texp, cp=cp)
        df['sabr Vol'] = sabrmodel.vol_for_price(strike, spot, texp)
        df['delta'] = sabrmodel.delta_numeric(strike, spot, texp, cp=cp)
        df['gamma'] = sabrmodel.gamma_numeric(strike, spot, texp, cp=cp)
        df['vega'] = sabrmodel.vega_numeric(strike, spot, texp, cp=cp)
        df['theta'] = sabrmodel.theta_numeric(strike, spot, texp, cp=cp)
        
        return df
    
    spot = etf.loc[etf['time'] == list(set(df['time']))[0], 'close'].values[0]
    texp = df['time_to_mature'].values[0]/250
    call = df.loc[df['type'] == 'C', :]
    put = df.loc[df['type'] == 'P', :]
    
    sabr_c = fit_split_cp(call, 1, spot, texp)
    sabr_p = fit_split_cp(put, -1, spot, texp)
    
    return pd.concat([sabr_c, sabr_p]).reset_index(drop = True)

# 2022-06-07 13:15:00 brentq报错 valueerror
insample = option2207.loc[option2207['time_to_mature'] > 37, :]
insample = insample.groupby('time').progress_apply(pandas_apply_sabr)

insample = insample.reset_index(drop = True)
insample = insample.loc[(np.abs(insample['delta']) < 0.9)&(np.abs(insample['delta']) > 0.1), :]
insample['signal'] = (insample['black Vol'] - insample['sabr Vol'])/insample['sabr Vol']
insample.dropna(inplace = True)

etf_insample = etf.set_index('time').reindex(insample.set_index('time').index.drop_duplicates()).reset_index()


# just a demo
# hedgemodel = path + '/HedgingModel/hedge.py'
# os.system('python '+hedgemodel)

from HedgingModel import hedge

pcpl = insample.groupby('time').apply(hedge.Put_Call_Parity_l).reset_index(drop = True)

pcpl1 = hedge.Hedge_Transform(pcpl, etf_insample)
hedge.Global_Exist(pcpl1)
pcpl1 = pcpl1.groupby('time').apply(hedge.Option_Position_Hold).reset_index(drop = True)
hedge.Global_Exist(pcpl1)
pcpl1 = pcpl1.groupby('time').apply(hedge.Hedge_ATM).reset_index(drop = True)

pcpl1.to_csv("/Users/lenovo/git_intro/SabrSummerProject2022/backtest/strategy_file/strategy_test.csv")






