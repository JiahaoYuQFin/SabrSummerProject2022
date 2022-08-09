# -*- coding: utf-8 -*-
"""
Created on 2022/8/4 13:45
@author: jhyu
"""
from OptionModel.sabr import SabrHagan2002
import pandas as pd
import numpy as np


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
        df_opt_new = df_opt_merge.groupby(['time'], as_index=False).apply(
            lambda x: window_of_greek(x)).set_index(['time', 'code'])
        return df_opt_new


def window_of_greek_onesurface(df: pd.DataFrame, nearest_num=3):
    model = SabrHagan2002(sigma=0.2, beta=0.5, intr=0.0278, divr=0.0195)

    df_new = df.iloc[(df['strike']-df['close_stock']).abs().argsort()].copy(deep=True)
    idx_call = (df_new['type'] == 1)
    idx_put = (df_new['type'] == -1)
    return df_new


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
    return df_new


def execute():
    etf_path = r'../03_data/300etf.pkl'
    option_path = r'../03_data/300etf_option2208.pkl'
    df_etf = pd.read_pickle(etf_path)
    df_opt = pd.read_pickle(option_path)
    model = OptionStra(df_opt=df_opt.query("time>='2022-07-06 09:45:00'"), df_stock=df_etf)
    res = model.get_greeks()


if __name__ == '__main__':
    execute()
