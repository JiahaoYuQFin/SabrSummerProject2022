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

def Option_Split_Rank(df, by_rank = 1, symmetry = True):
    '''
    use rank to choose options we want to long or short
    if not symmetry, choose by absolute value
    '''
    
    df['profit_position'] = 0
    if not symmetry:
        df.loc[np.abs(df['signal']).nlargest(by_rank).index, 'profit_position'] = -1
        df['profit_position'] = df['profit_position'] * np.sign(df['signal'])
    else:
        df.loc[df.nlargest(by_rank, 'signal').index, 'profit_position'] = -1
        df.loc[df.nsmallest(by_rank, 'signal').index, 'profit_position'] = 1
        
    return df

def Option_Split_Absolute(df, bar = 1e-3):
    '''
    use absolute price spread to choose options we want to long or short
    '''
    df.loc[df['signal'] > bar, 'profit_position'] = -1
    df.loc[df['signal'] < -bar, 'profit_position'] = 1
        
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
    df['position_available'] = np.abs(df['volume'] * quota)
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
def Option_Position_Add(df, quota = 0.5, trade_limit = 0.05, contract_limit = 0.2, notional_limit = 1e8):
    global exist_position
    
    if (df['profit_position']!=0).sum() == 0:
        exist_position = df.copy()
        return df
    
    spot = df.loc[df['type'] == 'S', 'close'].values[0]
    exist_position = exist_position.rename(columns = {'profit_position':'current_position'})
    df = pd.merge(df, exist_position[['code', 'current_position']], how = 'left', on = 'code')
    df['position_available'] = np.abs(df['volume'] * quota)
    df.loc[(df['profit_position']*df['current_position'])<=0, 'current_position'] = 0
    limit_position = notional_limit * trade_limit/spot/10000
    df['position_trade'] = limit_position - np.abs(df['current_position'])
    df['position_available'] = df[['position_trade', 'position_available']].min(axis=1)
    df['position_available'] = df['position_available'].apply(lambda x: max(x, 0))
    df['profit_position'] = df['profit_position'].values * df['position_available'] + df['current_position']
    df['profit_position'] = df['profit_position'].fillna(0)
    df['profit_position'] = df['profit_position'].astype(int)
    df = df.drop(columns=['position_available', 'current_position', 'position_trade'])
    
    exist_position = df.copy()
    
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

def Hedge_ATM(df, bar = 1.5e-4, delta_tolerance = 0):
    '''
    use more accurately priced option to hedge
    delta_tolerance: if exceed delta tolerance, then hedge delta to zero
    '''
    global exist_position
    
    spot = df.loc[df['type'] == 'S', 'close'].values[0]
    exist_position = exist_position.rename(columns = {'hedge_position':'current_position'})
    df = pd.merge(df, exist_position[['code', 'current_position']], how = 'left', on = 'code')
    df['hedge_position'] = df['current_position']
    df = df.drop(columns = 'current_position')
    profit_delta = np.sum(df['profit_position'] * df['delta'])
    df.loc[df['hedge_position']*df['profit_position'] != 0, 'hedge_position'] = 0
    df.loc[df['hedge_position']*df['delta']*profit_delta > 0, 'hedge_position'] = 0
    delta = profit_delta + np.sum(df['hedge_position'] * df['delta'])
    if abs(delta) > delta_tolerance:
        if delta < 0:
            df.loc[df['type'] == 'S', 'hedge_position'] = -round(delta)
        else:
            instruments = df.loc[(df['profit_position'] == 0)&(np.abs(df['signal']) < bar)&(df['type'] != 'S'), :]
            if len(instruments) == 0:
                df.loc[df['type'] == 'S', 'hedge_position'] = -round(delta)
            else:
                #print(np.abs(instruments['strike price'] - spot).idxmin())
                instrument_code = instruments.loc[np.abs(instruments['strike price'] - spot).idxmin(), 'code']
                df.loc[df['code'] == instrument_code, 'hedge_position'] = -round(delta/df.loc[df['code'] == instrument_code, 'delta'])
    
    exist_position = df.copy()
    
    return df

def Hedge_Vega(df, bar = 1.5e-4, delta_tolerance = 0, vega_tolerance = 0):
    '''
    use more accurately priced option to hedge, take control of vega
    delta_tolerance: if exceed delta tolerance, then hedge delta to zero
    '''
    global exist_position
    
    spot = df.loc[df['type'] == 'S', 'close'].values[0]
    exist_position = exist_position.rename(columns = {'hedge_position':'current_position'})
    df = pd.merge(df, exist_position[['code', 'current_position']], how = 'left', on = 'code')
    df['hedge_position'] = df['current_position']
    df = df.drop(columns = 'current_position')
    profit_delta = np.sum(df['profit_position'] * df['delta'])
    df.loc[df['hedge_position']*df['profit_position'] != 0, 'hedge_position'] = 0
    df.loc[df['hedge_position']*df['delta']*profit_delta > 0, 'hedge_position'] = 0
    delta = profit_delta + np.sum(df['hedge_position'] * df['delta'])
    vega = np.sum(df['profit_position'] * df['vega']) + np.sum(df['hedge_position'] * df['vega'])
    if (abs(delta) > delta_tolerance)&(abs(vega) > vega_tolerance):
        instruments = df.loc[(df['profit_position'] == 0)&(np.abs(df['signal']) < bar), :]
        if len(instruments) == 1:
            df.loc[df['type'] == 'S', 'hedge_position'] = -round(delta)
        else:
            con_eq = {'type':'eq',
                      'fun': lambda x: np.array(np.sum(x * instruments['delta'])+profit_delta)}
            con_ineq = {'type':'ineq',
                        'fun': lambda x: np.array(x[-1])}
            def func_vega(x):
                return (np.sum(x * instruments['vega']))**2
            x0 = np.zeros(len(instruments))
            res = minimize(func_vega, x0, constraints = (con_eq, con_ineq), method = 'SLSQP')
            print(res.message)
            df.loc[(df['profit_position'] == 0)&(np.abs(df['signal']) < bar), 'hedge_position'] = res.x           
    elif (abs(delta) > delta_tolerance)&(abs(vega) < vega_tolerance):
        if delta < 0:
            df.loc[df['type'] == 'S', 'hedge_position'] = -round(delta)
        else:
            instruments = df.loc[(df['profit_position'] == 0)&(np.abs(df['signal']) < bar)&(df['type'] != 'S'), :]
            if len(instruments) == 0:
                df.loc[df['type'] == 'S', 'hedge_position'] = -round(delta)
            else:
                #print(np.abs(instruments['strike price'] - spot).idxmin())
                instrument_code = instruments.loc[np.abs(instruments['strike price'] - spot).idxmin(), 'code']
                df.loc[df['code'] == instrument_code, 'hedge_position'] = -round(delta/df.loc[df['code'] == instrument_code, 'delta']) 
        
    
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

def Hedge_Optimize(df, bar, greek, delta_tolerance = 0):
    '''
    minimize theta, while keep delta close to zero
    '''
    global exist_position
    
    df = df.drop(columns = 'hedge_position')
    df = pd.merge(df, exist_position[['code', 'hedge_position']], how = 'left', on = 'code')
    df.loc[df['profit_position'] != 0, 'hedge_position'] = 0
    df = df.fillna(0)
    
    instruments = df.loc[df['profit_position'] == 0, :]
    instruments = df.loc[np.abs(df['signal']) < bar, :]
    delta = np.sum(df['profit_position'] * df['delta']) + np.sum(df['hedge_position'] * df['delta'])
    con_eq = {'type':'eq',
              'fun': lambda x: np.array(np.sum(x * instruments['delta'])+np.sum(df['profit_position'] * df['delta']))}
    con_ineq = {'type':'ineq',
                'fun': lambda x: np.array(x[-1])}
    
    def func_theta(x):
        return np.sum(x * instruments['theta'])
    
    def func_vega(x):
        return (np.sum(x * instruments['vega']))**2
    
    if abs(delta) > delta_tolerance:
        if greek == 'theta':
            x0 = np.zeros(len(instruments))
            res = minimize(func_theta, x0, constraints = (con_eq, con_ineq), method = 'SLSQP')
        elif greek == 'vega':
            x0 = np.zeros(len(instruments))
            res = minimize(func_vega, x0, constraints = (con_eq, con_ineq), method = 'SLSQP')
    else:
        exist_position = df
        return df
    
    print(res.message)
    print(res.x)
    df.loc[(df['profit_position'] == 0)&(np.abs(df['signal']) < bar), 'hedge_position'] = res.x
    exist_position = df
    
    return df










