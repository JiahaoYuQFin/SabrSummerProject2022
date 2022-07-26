#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:40:10 2022

@author: chenwynn
"""
import pandas as pd
import numpy as np

'''
特殊期权组合
'''

def Put_Call_Parity_l(df, by_rank = 1):
    '''
    long option put call parity
    if by_rank = 100(a relatively large number), trade all put call parities that satify condition
    '''
    parity = []
    for strike in df['strike price']:
        if df.loc[df['strike price'] == strike, 'signal'].prod() <= 0:
            parity.append(df.loc[df['strike price'] == strike, :])
    parity = pd.concat(parity)
    parity_c = (parity.loc[parity['type'] == 'call', :]).loc[parity['signal'] < 0, :]
    parity_p = (parity.loc[parity['type'] == 'put', :]).loc[parity['signal'] > 0, :]
    if by_rank <= len(parity_c):
        parity_c.loc[parity_c['signal'].nsmallest(by_rank), 'profit_position'] = 1
        for same_strike in parity_c.loc[parity_c['profit_position'] != 0, 'strike price']:
            parity_p.loc[parity_p['strike price'] == same_strike, 'profit_position'] = -1
    else:
        parity_c['profit_position'] = 1
        parity_p['profit_position'] = -1
    parity_l = pd.concat([parity_c, parity_p])
    
    return pd.merge(df, parity_l[['time', 'code', 'profit_position']], on = ['time', 'code'])

def Put_Call_Parity_s(df, by_rank = 1):
    '''
    short option put call parity
    if by_rank = 100(a relatively large number), trade all put call parities that satify condition
    '''
    parity = []
    for strike in df['strike price']:
        if df.loc[df['strike price'] == strike, 'signal'].prod() <= 0:
            parity.append(df.loc[df['strike price'] == strike, :])
    parity = pd.concat(parity)
    parity_c = (parity.loc[parity['type'] == 'call', :]).loc[parity['signal'] > 0, :]
    parity_p = (parity.loc[parity['type'] == 'put', :]).loc[parity['signal'] < 0, :]
    if by_rank <= len(parity_c):
        parity_p.loc[parity_p['signal'].nsmallest(by_rank), 'profit_position'] = 1
        for same_strike in parity_p.loc[parity_p['profit_position'] != 0, 'strike price']:
            parity_c.loc[parity_c['strike price'] == same_strike, 'profit_position'] = -1
    else:
        parity_c['profit_position'] = -1
        parity_p['profit_position'] = 1
    parity_s = pd.concat([parity_c, parity_p])
    
    return pd.merge(df, parity_s[['time', 'code', 'profit_position']], on = ['time', 'code'])
    
    
def Straddle(df, by_rank = 1):
    '''
    construct option straddle
    if by_rank = False, trade all straddles that satisfy condition
    '''
    if by_rank:
        df.loc[np.abs(df['signal']).nlargest(by_rank).index, 'profit_position'] = -1
        for strike in df.loc[df['profit_position'] != 0, 'strike price']:
            if df.loc[df['strike price'] == strike, 'signal'].prod() >= 0:
                df.loc[df['strike price'] == strike, 'profit_position'] == -1
                df.loc[df['strike price'] == strike, 'profit_position'] = df.loc[df['strike price'] == strike, 'profit_position'] * np.sign(df.loc[df['strike price'] == strike, 'signal'])
            else:
                df.loc[df['strike price'] == strike, 'profit_position'] == 0
    else:
        for strike in df['strike price']:
            if df.loc[df['strike price'] == strike, 'signal'].prod() >= 0:
                df.loc[df['strike price'] == strike, 'profit_position'] == -1
                df.loc[df['strike price'] == strike, 'profit_position'] = df.loc[df['strike price'] == strike, 'profit_position'] * np.sign(df.loc[df['strike price'] == strike, 'signal'])
    
    return df


def Bull_Call_Spread(df):
    '''
    construct option bull call spread
    '''
    call = df.loc[df['type'] == 'call', :]
    call_long = call.loc[call['signal'] < 0, :]
    call_short = call.loc[call['signal'] > 0, :]
    k1 = call_short['strike price'].min()
    k2 = call_long['strike price'].max()
    call_long = call_long.loc[call_long['strike price'] < k1, :]
    call_short = call_short.loc[call_short['strike price'] > k2, :]
    call_ls = pd.concat([call_long.loc[call_long['signal'].idxmin(), :], call_short.loc[call_short['signal'].idxmax(), :]])
    call_ls['profit_position'] = np.sign(call_ls['signal'])
    
    return pd.merge(df, call_ls[['time', 'code', 'profit_position']], on = ['time', 'code'])


def Bull_Put_Spread(df):
    '''
    construct option bull put spread
    '''
    return df

def Bear_Call_Spread(df):
    '''
    construct option bear call spread
    '''
    return df
    
    
def Bear_Put_Spread(df):
    '''
    construct option bear put spread
    '''
    return df
    


def Option_Split_Rank(df, by_rank = 1, symmetry = True):
    '''
    use rank to choose options we want to long or short
    if not symmetry, choose by absolute value
    '''
    
    '''
    if 'IV Deviation' in df.columns:
        pass
    else:
        df['IV Deviation'] = (df['black IV'] - df['sabr IV'])/df['sabr IV']
    '''
    
    df['profit_position'] = 0
    if not symmetry:
        df.loc[np.abs(df['signal']).nlargest(by_rank).index, 'profit_position'] = -1
        df['profit_position'] = df['profit_position'] * np.sign(df['signal'])
    else:
        df.loc[df.nlargest(by_rank, 'signal').index, 'profit_position'] = -1
        df.loc[df.nsmallest(by_rank, 'signal').index, 'profit_position'] = 1
        
    return df
        
'''
控制开仓数量
'''
# df must contain the calculation of greeks, profit_position, volume of each contract
# fluctuate with market volume
def Option_Position(df, quota = 0.5, limit = 1e6):
    df['position_available'] = np.abs(df['profit_position'] * df['volume'] * quota)
    position = df.loc[df['profit_position'] != 0, :].groupby('time')['position_available'].min()
    position = position.reset_index()
    df = pd.merge(df.drop(columns = 'position_available'), position, on = 'time')
    df['profit_position'] = df['profit_position'] * df['position_available']
    
    return df.drop(columns  = 'position_available')

# add position gradually


'''
对冲部分
'''

'''
before use functions below, plz add these codes

df['hedge_position'] = 0
time0 = df['time'].values[0]
exist_position = df.loc[df['time'] == time0, :]
'''

def Hedge_Spot(df, delta_tolerance = 0):
    '''
    use underlying spot to hedge

    '''
    
    global exist_position
    
    delta = (df['profit_position'] * df['delta']).sum()
    if np.abs(delta - exist_position['hedge_position'].sum()) > delta_tolerance:
        df.loc[df['type']=='spot', 'hedge_position'] = -delta
    else:
        df.loc[df['type']=='spot', 'hedge_position'] = exist_position['hedge_position'].sum()
    
    exist_position = df
    
    return df


def Hedge_ATM(df, atm = 0.2, delta_tolerance = 0):
    '''
    use ATM option to hedge, atm parameter is used to define close to ATM options
    delta_tolerance: if exceed delta tolerance, then hedge delta to zero
    '''
    global exist_position
    
    df['hedge_position'] = exist_position['hedge_position']
    df.loc[df['hedge_position']*df['profit_position'] != 0, 'hedge_position'] = 0
    df.loc[np.abs(df.loc[df['hedge_position'] != 0, 'delta'] - 0.5) > atm, 'hedge_position'] = 0
    delta = ((df['hedge_position'] + df['profit_position']) * df['delta']).sum()
    hedge_delta = (df['hedge_position'] * df['delta']).sum()
    if np.abs(hedge_delta) < np.abs(delta):
        df['hedge_position'] = 0
    delta = ((df['hedge_position'] + df['profit_position']) * df['delta']).sum()
    if np.abs(delta) < delta_tolerance:
        pass
    else:
        instruments = df.loc[delta/df['delta']<0, :]
        hedge_option = np.abs(instruments['strike price'] - instruments['spot']).argmin()
        df.loc[hedge_option, 'hedge_position'] = delta/df.loc[hedge_option, 'delta']
    
    exist_position = df
    
    return df

from scipy.optimize import minimize

def Hedge_Optimize_Theta(df, long_spot = True, delta_tolerance = 0):
    '''
    minimize theta, while keep delta close to zero
    '''
    
    instruments = df.loc[df['profit_position'] == 0, :]
    con_eq = {'type':'eq',
              'fun': lambda x: np.array(np.sum(x * instruments['delta'])+np.sum(df['profit_position'] * df['delta']))}
    con_ineq = {'type':'ineq',
                'fun': lambda x: np.array(x[-1])}
    con_tole = {'type':'ineq',
                'fun': lambda x: np.array([np.sum(x * instruments['delta'])+np.sum(df['profit_position'] * df['delta'])-delta_tolerance,
                                           -np.sum(x * instruments['delta'])+np.sum(df['profit_position'] * df['delta'])+delta_tolerance])}
    x0 = np.zeros(len(instruments))
    
    def func_theta(x):
        return np.sum(x * instruments['theta'])
    
    if delta_tolerance == 0:
        if long_spot:
            res = minimize(func_theta, x0, constraints = (con_eq, con_ineq), method = 'SLSQP')
        else:
            res = minimize(func_theta, x0, constraints = (con_eq), method = 'SLSQP')
    else:
        if long_spot:
            res = minimize(func_theta, x0, constraints = (con_tole, con_ineq), method = 'SLSQP')
        else:
            res = minimize(func_theta, x0, constraints = (con_tole), method = 'SLSQP')
    
    df.loc[df['profit_position'] == 0, 'hedge_position'] = res.x
    
    return df

