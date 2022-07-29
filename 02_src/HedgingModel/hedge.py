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
    for strike in set(df['strike price']):
        if (np.sum(df['strike price'] == strike) == 2) and (df.loc[df['strike price'] == strike, 'signal'].prod() <= 0):
            parity.append(df.loc[df['strike price'] == strike, :])
    if parity == []:
        df['profit_position'] = 0
        return df
    parity = pd.concat(parity)
    parity_c = parity.loc[parity['type'] == 'C', :]
    parity_c = parity_c.loc[parity_c['signal'] < 0, :]
    parity_p = parity.loc[parity['type'] == 'P', :]
    parity_p = parity_p.loc[parity_p['signal'] > 0, :]
    if by_rank <= len(parity_c):
        parity_c.loc[parity_c['signal'].nsmallest(by_rank).index, 'profit_position'] = 1
        for same_strike in parity_c.loc[parity_c['profit_position'] == 1, 'strike price']:
            parity_p.loc[parity_p['strike price'] == same_strike, 'profit_position'] = -1
    else:
        parity_c['profit_position'] = 1
        parity_p['profit_position'] = -1
    parity_l = pd.concat([parity_c, parity_p])
    
    return pd.merge(df, parity_l[['time', 'code', 'profit_position']], on = ['time', 'code'], how = 'left').reset_index(drop = True)

def Put_Call_Parity_s(df, by_rank = 1):
    '''
    short option put call parity
    if by_rank = 100(a relatively large number), trade all put call parities that satify condition
    '''
    parity = []
    for strike in set(df['strike price']):
        if (np.sum(df['strike price'] == strike) == 2) and (df.loc[df['strike price'] == strike, 'signal'].prod() <= 0):
            parity.append(df.loc[df['strike price'] == strike, :])
    if parity == []:
        df['profit_position'] = 0
        return df
    parity = pd.concat(parity)
    parity_c = parity.loc[parity['type'] == 'C', :]
    parity_c = parity_c.loc[parity_c['signal'] > 0, :]
    parity_p = parity.loc[parity['type'] == 'P', :]
    parity_p = parity_p.loc[parity_p['signal'] < 0, :]
    if by_rank <= len(parity_c):
        parity_p.loc[parity_p['signal'].nsmallest(by_rank).index, 'profit_position'] = 1
        for same_strike in parity_p.loc[parity_p['profit_position'] == 1, 'strike price']:
            parity_c.loc[parity_c['strike price'] == same_strike, 'profit_position'] = -1
    else:
        parity_c['profit_position'] = -1
        parity_p['profit_position'] = 1
    parity_s = pd.concat([parity_c, parity_p])
    
    return pd.merge(df, parity_s[['time', 'code', 'profit_position']], on = ['time', 'code'], how = 'left').reset_index(drop = True)
    
    
def Straddle(df, by_rank = 1):
    '''
    construct option straddle
    if by_rank = False, trade all straddles that satisfy condition
    '''
    if by_rank:
        df.loc[np.abs(df['signal']).nlargest(by_rank).index, 'profit_position'] = -1
        for strike in df.loc[df['profit_position'] != 0, 'strike price']:
            if df.loc[df['strike price'] == strike, 'signal'].prod() >= 0:
                df.loc[df['strike price'] == strike, 'profit_position'] = -1
                df.loc[df['strike price'] == strike, 'profit_position'] = df.loc[df['strike price'] == strike, 'profit_position'] * np.sign(df.loc[df['strike price'] == strike, 'signal'])
            else:
                df.loc[df['strike price'] == strike, 'profit_position'] = 0
    else:
        for strike in df['strike price']:
            if df.loc[df['strike price'] == strike, 'signal'].prod() >= 0:
                df.loc[df['strike price'] == strike, 'profit_position'] = -1
                df.loc[df['strike price'] == strike, 'profit_position'] = df.loc[df['strike price'] == strike, 'profit_position'] * np.sign(df.loc[df['strike price'] == strike, 'signal'])
    
    return df


def Bull_Call_Spread(df, by_rank = 1):
    '''
    construct option bull call spread
    '''
    call = df.loc[df['type'] == 'C', :]
    call_long = call.loc[call['signal'] < 0, :]
    call_short = call.loc[call['signal'] > 0, :]
    k1 = call_short['strike price'].min()
    k2 = call_long['strike price'].max()
    call_long = call_long.loc[call_long['strike price'] < k1, :]
    call_short = call_short.loc[call_short['strike price'] > k2, :]
    call_ls = pd.concat([call_long.loc[call_long['signal'].nsmallest(by_rank).index, :], call_short.loc[call_short['signal'].nlargest(by_rank).index, :]])
    call_ls['profit_position'] = np.sign(call_ls['signal'])
    
    return pd.merge(df, call_ls[['time', 'code', 'profit_position']], on = ['time', 'code'], how = 'left').reset_index(drop = True)


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
before use functions below, plz use this function
'''

def Hedge_Transform(df, spot):
    spot['type'] = 'S'
    spot['delta'] = 1
    df = pd.concat([df, spot])
    df['hedge_position'] = 0
    df = df.fillna(0)
    
    return df    

def Global_Exist(df):
    global exist_position
    
    time0 = df['time'].sort_values().values[0]
    exist_position = df.loc[df['time'] == time0, :]
    exist_position['profit_position'] = 0
    
    return 0
    

'''
控制开仓数量
'''
# df must contain the calculation of greeks, profit_position, volume of each contract
# fluctuate with market volume
def Option_Position_Hold(df, quota = 0.5, contract_limit = 0.2, notional_limit = 1e8):
    global exist_position
    
    if (df['profit_position']!=0).sum() == 0:
        exist_position = df.copy()
        return df
    
    spot = df.loc[df['type'] == 'S', 'close'].values[0]
    exist_position = exist_position.rename(columns = {'profit_position':'current_position'})
    df = pd.merge(df, exist_position[['code', 'current_position']], how = 'left', on = 'code')
    df['position_available'] = np.abs(df['profit_position'] * df['volume'] * quota)
    position = df.loc[df['profit_position'] != 0, 'position_available'].min() 
    notional_position = notional_limit/spot/10000 - np.abs(df.loc[(df['profit_position']*df['current_position'])>0, 'current_position']).sum()
    limit_position = min(notional_position, notional_limit * contract_limit/spot/10000)
    position = min(position, limit_position)
    df['profit_position'] = df['profit_position'] * position
    df['profit_position'] = df['profit_position'].astype(int)
    df.loc[(df['profit_position']*df['current_position'])>0, 'profit_position'] = df.loc[(df['profit_position']*df['current_position'])>0, 'current_position']
    df = df.drop(columns=['position_available', 'current_position'])
    
    exist_position = df.copy()
    
    return df

# add position gradually
def Option_Position_Add(df, quota = 0.2, limit = 1e6):
    
    return df

'''
对冲部分
'''
    
    

def Hedge_Spot(df, delta_tolerance = 0):
    '''
    use underlying spot to hedge

    '''
    
    global exist_position
    
    delta = (df['profit_position'] * df['delta']).sum()
    if np.abs(delta - exist_position['hedge_position'].sum()) > delta_tolerance:
        df.loc[df['type']=='S', 'hedge_position'] = -round(delta)
    else:
        df.loc[df['type']=='S', 'hedge_position'] = exist_position['hedge_position'].sum()
    
    exist_position = df.copy()
    
    return df


def Hedge_ATM_theta(df, atm = 0.2, delta_tolerance = 0):
    '''
    use ATM option to hedge, atm parameter is used to define close to ATM options
    delta_tolerance: if exceed delta tolerance, then hedge delta to zero
    '''
    global exist_position
    
    spot = df.loc[df['type'] == 'S', 'close'].values[0]
    exist_position = exist_position.rename(columns = {'hedge_position':'current_position'})
    df = pd.merge(df, exist_position[['code', 'current_position']], how = 'left', on = 'code')
    df['hedge_position'] = df['current_position']
    df = df.drop(columns = 'current_position')
    df.loc[df['hedge_position']*df['profit_position'] != 0, 'hedge_position'] = 0
    df.loc[np.abs(df['delta'] - 0.5) > atm, 'hedge_position'] = 0
    delta = (df['profit_position'] * df['delta']).sum()
    hedge_delta = (df['hedge_position'] * df['delta']).sum()
    if np.sign(hedge_delta) != np.sign(delta):
        df['hedge_position'] = 0
    delta = ((df['hedge_position'] + df['profit_position']) * df['delta']).sum()
    if np.abs(delta) <= delta_tolerance:
        pass
    else:
        instruments = df.loc[delta/df['delta']<0, :]
        hedge_option = np.abs(instruments['strike price'] - spot).argmin()
        df.loc[hedge_option, 'hedge_position'] = -round(delta/df.loc[hedge_option, 'delta'])
    
    exist_position = df.copy()
    
    return df

def Hedge_ATM_vega(df, atm = 0.2, delta_tolerance = 0):
    '''
    use ATM option to hedge, atm parameter is used to define close to ATM options
    delta_tolerance: if exceed delta tolerance, then hedge delta to zero
    '''
    global exist_position
    
    spot = df.loc[df['type'] == 'S', 'close'].values[0]
    exist_position = exist_position.rename(columns = {'hedge_position':'current_position'})
    df = pd.merge(df, exist_position[['code', 'current_position']], how = 'left', on = 'code')
    df['hedge_position'] = df['current_position']
    df = df.drop(columns = 'current_position')
    df.loc[df['hedge_position']*df['profit_position'] != 0, 'hedge_position'] = 0
    df.loc[np.abs(df['delta'] - 0.5) > atm, 'hedge_position'] = 0
    delta = (df['profit_position'] * df['delta']).sum()
    hedge_delta = (df['hedge_position'] * df['delta']).sum()
    if np.sign(hedge_delta) != np.sign(delta):
        df['hedge_position'] = 0
    delta = ((df['hedge_position'] + df['profit_position']) * df['delta']).sum()
    vega = ((df['hedge_position'] + df['profit_position']) * df['vega']).sum()
    if np.abs(delta) <= delta_tolerance:
        pass
    else:
        instruments = df.loc[vega/df['vega']<0, :]
        hedge_option = np.abs(instruments['strike price'] - spot).argmin()
        df.loc[hedge_option, 'hedge_position'] = -round(delta/df.loc[hedge_option, 'delta'])
    
    exist_position = df.copy()
    
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










