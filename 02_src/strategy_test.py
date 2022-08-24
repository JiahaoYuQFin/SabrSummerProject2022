# -*- coding: utf-8 -*-
"""
Created on 2022/8/4 13:45
@author: jhyu
"""
from OptionModel.sabr import SabrHagan2002
import pandas as pd
import numpy as np
import scipy.optimize as sopt


class OptionStra:
    def __init__(self, df_stock: pd.DataFrame, df_opt: pd.DataFrame):
        self.df_stock = df_stock
        self.df_opt = df_opt

    def get_greeks(self):
        df_opt = self.df_opt.copy(deep=True)
        df_stock = self.df_stock.copy(deep=True)
        df_opt['texp'] = (df_opt['maturity_date']-df_opt['time']).dt.total_seconds()/3600/24/365
        df_opt_merge = df_opt.merge(df_stock[['time', 'close']], suffixes=('', '_stock'),
                                    left_on=['time'], right_on=['time'],
                                    how='left').ffill()#.set_index(['code'])

        # sabrmodel = SabrHagan2002(sigma=0.2, beta=0.5, intr=0.0278, divr=0.0195)
        df_opt_new = (df_opt_merge.groupby(['time'], as_index=False).apply(
            lambda x: window_of_greek(x))).set_index(['time', 'code'])
        return df_opt_new

    def straddle(self, df_greeks: pd.DataFrame) -> pd.DataFrame:
        df_pos = df_greeks.reset_index().groupby('time', as_index=False).apply(lambda x: window_of_straddle(x))
        df_pos['position'] = df_pos.groupby('code')['position'].shift(1).fillna(0)
        df_pos = df_pos.set_index(['time', 'code'])
        return df_pos[['open', 'close', 'volume', 'delta', 'gamma', 'theta', 'vega', 'volga', 'position']]

    def volga_stra(self, df_greeks: pd.DataFrame, direction='long') -> pd.DataFrame:
        if direction == 'long':
            df_pos = df_greeks.reset_index().groupby('time', as_index=False).apply(lambda x: window_of_volga(x, 1))
        else:
            df_pos = df_greeks.reset_index().groupby('time', as_index=False).apply(lambda x: window_of_volga(x, -1))
        df_pos['position'] = df_pos.groupby('code')['position'].shift(1).fillna(0)
        df_pos = df_pos.set_index(['time', 'code'])
        return df_pos[['open', 'close', 'volume', 'delta', 'gamma', 'theta', 'vega', 'volga', 'position']]

    def theta_stra(self, df_greeks: pd.DataFrame, direction='long') -> pd.DataFrame:
        if direction == 'long':
            df_pos = df_greeks.reset_index().groupby('time', as_index=False).apply(lambda x: window_of_theta(x, 1))
        else:
            df_pos = df_greeks.reset_index().groupby('time', as_index=False).apply(lambda x: window_of_theta(x, -1))
        df_pos['position'] = df_pos.groupby('code')['position'].shift(1).fillna(0)
        df_pos = df_pos.set_index(['time', 'code'])
        return df_pos[['open', 'close', 'volume', 'delta', 'gamma', 'theta', 'vega', 'volga', 'position']]


def window_of_straddle(df: pd.DataFrame, num: int = 1):
    assert 0 < num < 5
    df_pool = df.query("((type==1)&(0.1<delta<0.9))|((type==-1)&(-0.9<delta<-0.1))").copy(deep=True)
    texp = df_pool['texp'].unique()
    target_texp = texp[0] if (texp[0] >= (2.5 / 24 / 365)) & ((df_pool['texp'] == texp[0]).sum() > 3) else texp[1]
    df_pool = df_pool[df_pool['texp'] == target_texp]
    strikes = df_pool['strike'].unique()
    for i in range(num):
        idx_strike = df_pool.query("strike==@strikes[@i]").index
        while idx_strike.size < 2:
            i += 1
            idx_strike = df_pool.query("strike==@strikes[@i]").index
        a = np.vstack([df_pool.loc[idx_strike, 'delta'].tolist(), [1, 1]])
        b = np.array([0, 1])
        x = np.linalg.solve(a, b)
        df.loc[idx_strike, 'position'] = x
    return df


def window_of_volga(df: pd.DataFrame, direction=1):
    df_pool = df.query("((type==1)&(0.1<delta<0.9))|((type==-1)&(-0.9<delta<-0.1))").copy(deep=True)
    greeks_arr = df_pool[['delta', 'gamma', 'vega', 'volga']].values
    cons = (
        {'type': 'eq', 'fun': lambda x: x.sum()-1},
        {'type': 'ineq', 'fun': lambda x: x + 1},
        {'type': 'ineq', 'fun': lambda x: 1 - x}
    )
    sol = sopt.minimize(fun=lambda x: -direction*x@greeks_arr[:, -1] + 0.15 * abs(x@greeks_arr[:, 0]) +
                                      0.15 * abs(x@greeks_arr[:, 2]) + 0.1 * abs(x@greeks_arr[:, 1]),
                        x0=np.zeros(greeks_arr.shape[0]), method='SLSQP', constraints=cons)
    df.loc[df_pool.index, 'position'] = sol.x
    return df


def window_of_theta(df: pd.DataFrame, direction=1):
    df_pool = df.query("((type==1)&(0.1<delta<0.9))|((type==-1)&(-0.9<delta<-0.1))").copy(deep=True)
    greeks_arr = df_pool[['delta', 'gamma', 'vega', 'theta']].values
    cons = (
        {'type': 'eq', 'fun': lambda x: x.sum()-1},
        {'type': 'ineq', 'fun': lambda x: x + 1},
        {'type': 'ineq', 'fun': lambda x: 1 - x}
    )
    sol = sopt.minimize(
        fun=lambda x: -direction * x @ greeks_arr[:, -1] + 0.15 * abs(x @ greeks_arr[:, 0]) + 0.06 * abs(
            x @ greeks_arr[:, 1]) + 0.06 * abs(x @ greeks_arr[:, 2]),
        x0=np.zeros(greeks_arr.shape[0]), method='SLSQP', constraints=cons)
    df.loc[df_pool.index, 'position'] = sol.x
    return df


def window_of_greek(df: pd.DataFrame, nearest_num=3):
    model = SabrHagan2002(sigma=0.2, beta=0.5, intr=0.0199, divr=0.0198)

    df_new = df.iloc[(df['strike']-df['close_stock']).abs().argsort()].copy(deep=True)

    idx_call = (df_new['type'] == 1)
    idx_put = (df_new['type'] == -1)
    cprice = df_new.loc[idx_call, 'close'].values
    cstrike = df_new.loc[idx_call, 'strike'].values
    pprice = df_new.loc[idx_put, 'close'].values
    pstrike = df_new.loc[idx_put, 'strike'].values
    spot = df_new['close_stock'].values[0]
    texp = df_new['texp'].values[0]
    df_new.loc[idx_call, 'bs_iv'] = model.impvol(price=cprice, strike=cstrike, spot=spot, texp=texp, cp=1)
    df_new.loc[idx_put, 'bs_iv'] = model.impvol(price=pprice, strike=pstrike, spot=spot, texp=texp, cp=-1)

    # drop options with bs_iv equal to NaN
    # idx_call_ivnotnull = df_new.query("type==1 & bs_iv.notna()").index
    # idx_put_ivnotnull = df_new.query("type==-1 & bs_iv.notna()").index
    # cprice = df_new.loc[idx_call_ivnotnull, 'close'].values
    # cstrike = df_new.loc[idx_call_ivnotnull, 'strike'].values
    # pprice = df_new.loc[idx_put_ivnotnull, 'close'].values
    # pstrike = df_new.loc[idx_put_ivnotnull, 'strike'].values

    # call
    call_arr = df_new.loc[idx_call, ['bs_iv', 'strike']].iloc[:nearest_num].sort_values('strike').values
    c3 = call_arr[[0, int(nearest_num/2), -1], :].copy()
    model.calibrate3(price_or_vol3=c3[:, 0], strike3=c3[:, 1], spot=spot, texp=texp,
                     cp=1, setval=True, is_vol=True)
    df_new.loc[idx_call, 'sabr_iv'] = model.vol_for_price(strike=cstrike, spot=spot, texp=texp)
    df_new.loc[idx_call, 'delta'] = model.delta(strike=cstrike, spot=spot, texp=texp, cp=1)
    df_new.loc[idx_call, 'gamma'] = model.gamma(strike=cstrike, spot=spot, texp=texp, cp=1)
    df_new.loc[idx_call, 'theta'] = model.theta(strike=cstrike, spot=spot, texp=texp, cp=1)
    df_new.loc[idx_call, 'vega'] = model.vega(strike=cstrike, spot=spot, texp=texp, cp=1)
    df_new.loc[idx_call, 'vanna'] = model.vanna_numeric(strike=cstrike, spot=spot, texp=texp, cp=1)
    df_new.loc[idx_call, 'volga'] = model.volga_numeric(strike=cstrike, spot=spot, texp=texp, cp=1)

    # put
    put_arr = df_new.loc[idx_put, ['bs_iv', 'strike']].iloc[:nearest_num].sort_values('strike').values
    p3 = put_arr[[0, int(nearest_num/2), -1], :].copy()
    model.calibrate3(price_or_vol3=p3[:, 0], strike3=p3[:, 1], spot=spot, texp=texp,
                     cp=-1, setval=True, is_vol=True)
    df_new.loc[idx_put, 'sabr_iv'] = model.vol_for_price(strike=pstrike, spot=spot, texp=texp)
    df_new.loc[idx_put, 'delta'] = model.delta(strike=pstrike, spot=spot, texp=texp, cp=-1)
    df_new.loc[idx_put, 'gamma'] = model.gamma(strike=pstrike, spot=spot, texp=texp, cp=-1)
    df_new.loc[idx_put, 'theta'] = model.theta(strike=pstrike, spot=spot, texp=texp, cp=-1)
    df_new.loc[idx_put, 'vega'] = model.vega(strike=pstrike, spot=spot, texp=texp, cp=-1)
    df_new.loc[idx_put, 'vanna'] = model.vanna_numeric(strike=pstrike, spot=spot, texp=texp, cp=-1)
    df_new.loc[idx_put, 'volga'] = model.volga_numeric(strike=pstrike, spot=spot, texp=texp, cp=-1)

    return df_new


def execute():
    etf_path = r'../03_data/300etf.pkl'
    option_path = r'../03_data/300etf_option2207.pkl'
    df_etf = pd.read_pickle(etf_path)
    df_opt = pd.read_pickle(option_path)
    df_greeks = pd.read_pickle(r'../03_data/greeks_22060708.pkl')
    # e
    model = OptionStra(df_opt=df_opt.query("time>='2022-06-16 09:45:00'"), df_stock=df_etf)
    # df_theta_pos = model.theta_stra(df_greeks=df_greeks, direction='long')
    # df_theta_pos.to_pickle(r'../backtest/strategy_file/strategy_theta.pkl')
    df_volga_l_pos = model.volga_stra(df_greeks=df_greeks, direction='long')
    df_volga_l_pos.to_pickle(r'../backtest/strategy_file/strategy_long_volga.pkl')
    # df_volga_s_pos = model.volga_stra(df_greeks=df_greeks, direction='short')
    # df_volga_s_pos.to_pickle(r'../backtest/strategy_file/strategy_short_volga.pkl')
    # df_straddle_pos.to_pickle(r'../backtest/strategy_file/strategy_straddle.pkl')


if __name__ == '__main__':
    execute()
