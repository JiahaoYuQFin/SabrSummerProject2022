#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 15:27:04 2022

@author: chenwynn
"""

import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt
import copy

def computeFirstDerivative(v_u_plus_du , v_u_minus_du , du):
    first_derivative = (v_u_plus_du - v_u_minus_du) / (2.0 * du)
    return first_derivative

def computeSecondDerivative(v_u, v_u_plus_du, v_u_minus_du, du):
    second_derivative = ((v_u_plus_du - 2.0*v_u + v_u_minus_du)/ (du * du)) 
    return second_derivative


class BlackScholes:
    def __init__(self, y, expiry, S, r, isCall):
        '''
        y : np.array, strike price
        expiry : float, date to maturity in terms of year
        S : int, spot price
        r: riskfree interest rate
        isCall: np.array, e.g. True if is call
        '''
        self.y = y
        self.expiry = expiry
        self.S = S
        self.r = r
        self.isCall = isCall
        
    def dPlusBlack(self, vol):
        d_plus = ((np.log(self.S / self.y) + (self.r + 0.5 * vol ** 2) * self.expiry) / vol / math.sqrt(self.expiry)) 
        return d_plus
    
    def dMinusBlack(self, vol):
        return self.dPlusBlack(vol) - vol * math.sqrt(self.expiry)
    
    def black(self, vol):
        '''
        

        Parameters
        ----------
        vol : np.array, vol
        
        Returns
        -------
        np.array, option black price

        '''
        option_value = np.ones(len(self.y))
        if self.expiry == 0.0:
            option_value[self.isCall] = np.maximum(self.S - self.y[self.isCall], 0.0)
            option_value[~self.isCall] = np.maximum(self.y[~self.isCall] - self.S, 0.0)
        else:
            d1 = self.dPlusBlack(vol = vol)
            d2 = self.dMinusBlack(vol = vol)
            option_value[self.isCall] = self.S * norm.cdf(d1[self.isCall]) - self.y[self.isCall] * math.exp(-self.r*self.expiry) * norm.cdf(d2[self.isCall])
            option_value[~self.isCall] = self.y[~self.isCall] * math.exp(-self.r*self.expiry) * norm.cdf(-d2[~self.isCall]) - self.S * norm.cdf(-d1[~self.isCall])
        
        return option_value
    
    def blackDelta(self, vol):
        delta = np.ones(len(self.y))
        delta[self.isCall] = norm.cdf(self.dPlusBlack(vol))[self.isCall]
        delta[~self.isCall] = norm.cdf(self.dMinusBlack(vol))[~self.isCall] - 1
        return delta
    
    def blackVega(self, vol):
        return self.y * norm.pdf(self.dPlusBlack(vol)) * math.sqrt(self.expiry)
    
    def blackIV(self, price, max_try = 100000, min_cent = 0.001):
        '''
        No explicit method, use newton method to approximate

        Parameters
        ----------
        price : np.array, option price

        Returns
        -------
        np.array, implied volatility using BS model

        '''
        _vol = np.ones(price.shape)/2
        for i in range(max_try):
            _bs_price = self.black(_vol)
            diff = price - _bs_price
            vega = self.blackVega(_vol)
            if np.nanmax(abs(diff)) < min_cent:
                return _vol
            _vol += diff/vega
        return _vol 
    
    

class SABR():
    def __init__(self, y, expiry, S, r, isCall):
        '''
        y : np.array, strike price
        expiry : float, date to maturity in terms of year
        S : int, spot price
        r: riskfree interest rate
        isCall: np.array, e.g. True if is call
        '''
        self.y = y
        self.expiry = expiry
        self.S = S
        self.r = r
        self.isCall = isCall
        self.BS = BlackScholes(y, expiry, S, r, isCall)
    
    
    def haganLogNormalApprox(self, BS, alpha, beta, nu, rho):
        '''
        

        Parameters
        ----------
        BS: class BlackScholes Object
        alpha : volatility 
        beta : cev
        nu: volatility of volatility
        rho: correlation factor, similar effect to beta

        Returns
        -------
        Implied volatility using Hagan et al. estimation

        '''
        F0 = BS.S * math.exp(BS.r * BS.expiry)
        zero_index = np.where(BS.y == F0)
        
        one_beta = 1.0 - beta
        one_betasqr = one_beta * one_beta
        fK = F0 * BS.y
        fK_beta = np.power(fK, one_beta / 2.0) 
        log_fK = np.log(F0 / BS.y)
        z = nu / alpha * fK_beta * log_fK
        x = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1 - rho))
        z[zero_index] = 1
        x[zero_index] = 1
        # nonzero = (z / x) * (1.0 + one_betasqr / 24.0 * log_fK * log_fK + np.power(one_beta * log_fK, 4) / 1920.0)
        sigma_l = (alpha / fK_beta / (1.0 + one_betasqr / 24.0 * log_fK * log_fK + np.power(one_beta * log_fK, 4) / 1920.0) * (z / x))
        sigma_exp = (one_betasqr / 24.0 * alpha * alpha / fK_beta / fK_beta + 0.25 * rho * beta * nu * alpha / fK_beta + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu)
        sigma = sigma_l * ( 1.0 + sigma_exp * BS.expiry) 
        
        return sigma
    
    def haganFit_volume(self, BS, price, atm_index, volume, start_point):
        '''
        volume: at least 1

        Returns
        -------
        Best parameters

        '''
        bvol = BS.blackIV(price)
        if True in np.isnan(bvol):
            return 'NaN in Black Implied Volatility Calculation'
        F0 = BS.S * math.exp(BS.r * BS.expiry)
        for strike in np.unique(BS.y):
            volume[BS. y == strike] = volume[BS. y == strike]/np.sum(volume[BS. y == strike])*100
        
        def fun(args):
            alpha, beta, nu, rho = args
            # alpha = bvol[atm_index] * F0 ** (1-beta)
            return ((bvol - self.haganLogNormalApprox(BS, alpha, beta, nu, rho)) ** 2 * np.log(volume)).sum()
        '''
        cons = ({'type': 'ineq', 'fun': lambda x: x[0]},\
                {'type': 'ineq', 'fun': lambda x: -x[0]+1},\
                {'type': 'ineq', 'fun': lambda x: x[2]+1},\
                {'type': 'ineq', 'fun': lambda x: -x[2]+1})
        '''
        b0 = (0.1, 0.5)
        b1 = (0, 1)
        b2 = (0, 10)
        b3 = (-1, 1)
        x0 = start_point
        res = minimize(fun, x0, method = 'SLSQP', bounds=(b0, b1, b2, b3))
        print(res.message)

        return res.x
    
    def haganFit_delta(self, BS, price, atm_index, start_point):
        '''

        Returns
        -------
        Best parameters

        '''
        bvol = BS.blackIV(price)
        bdelta = BS.blackDelta(bvol) * 100
        if True in np.isnan(bvol):
            return 'NaN in Black Implied Volatility Calculation'
        # F0 = BS.S * math.exp(BS.r * BS.expiry)
        
        def fun(args):
            alpha, beta, nu, rho = args
            # alpha = bvol[atm_index] * F0 ** (1-beta)
            return ((bvol - self.haganLogNormalApprox(BS, alpha, beta, nu, rho)) ** 2 * (100 - np.abs(bdelta))).sum()
        '''
        cons = ({'type': 'ineq', 'fun': lambda x: x[0]},\
                {'type': 'ineq', 'fun': lambda x: -x[0]+1},\
                {'type': 'ineq', 'fun': lambda x: x[2]+1},\
                {'type': 'ineq', 'fun': lambda x: -x[2]+1})
        '''
        b0 = (0.1, 0.5)
        b1 = (0, 1)
        b2 = (0, 10)
        b3 = (-1, 1)
        x0 = start_point
        res = minimize(fun, x0, method = 'SLSQP', bounds=(b0, b1, b2, b3))
        print(res.message)

        return res.x
    
    def haganFit_split(self, BS, price, atm_index, start_point):
        '''
        volume: at least 1

        Returns
        -------
        Best parameters

        '''
        bvol = BS.blackIV(price)
        bdelta = BS.blackDelta(bvol) * 100
        if True in np.isnan(bvol):
            return 'NaN in Black Implied Volatility Calculation'
        atm_vol = bvol[atm_index]
        spread = BS.y[atm_index] - BS.S
        if spread > 0:
            atm_arg = (BS.y > BS.y[atm_index]-0.2)&(BS.y < BS.y[atm_index]+0.1)
        else:
            atm_arg = (BS.y > BS.y[atm_index]-0.1)&(BS.y < BS.y[atm_index]+0.2)
        BS_atm = BlackScholes(BS.y[atm_arg], BS.expiry, BS.S, BS.r, BS.isCall[atm_arg])
        
        # F0 = BS.S * math.exp(BS.r * BS.expiry)
        
        def optimize_ls(BS, bvol, bdelta, start_point):
            def atm_fun(args):
                alpha, beta, nu, rho = args
                # alpha = atm_vol * F0 ** (1-beta)
                return ((bvol - self.haganLogNormalApprox(BS, alpha, beta, nu, rho)) ** 2 * (100 - np.abs(bdelta))).sum()
            '''
            cons = ({'type': 'ineq', 'fun': lambda x: x[0]},\
                    {'type': 'ineq', 'fun': lambda x: -x[0]+1},\
                    {'type': 'ineq', 'fun': lambda x: x[2]+1},\
                    {'type': 'ineq', 'fun': lambda x: -x[2]+1})
            '''
            b0 = (0.1, 0.5)
            b1 = (0, 1)
            b2 = (0, 10)
            b3 = (-1, 1)
            res = minimize(atm_fun, start_point, method = 'SLSQP', bounds=(b0, b1, b2, b3))
            return res
        
        x0 = np.array([0.2, 1, 2.8, -0.2])
        if len(np.unique(BS.y)) >= 4:
            res = optimize_ls(BS_atm, bvol[atm_arg], bdelta[atm_arg], x0)
            bvol[atm_arg] = self.haganLogNormalApprox(BS_atm, res.x[0], res.x[1], res.x[2], res.x[3])
            y_low = np.sort(np.unique(BS.y))[1]
            low_arg = (BS.y <= y_low)
            BS_low = BlackScholes(BS.y[low_arg], BS.expiry, BS.S, BS.r, BS.isCall[low_arg])
            res = optimize_ls(BS_low, bvol[low_arg], bdelta[low_arg], x0)
            bvol[low_arg] = self.haganLogNormalApprox(BS_low, res.x[0], res.x[1], res.x[2], res.x[3])
            y_high = np.sort(np.unique(BS.y))[-2]
            high_arg = (BS.y >= y_high)
            BS_high = BlackScholes(BS.y[high_arg], BS.expiry, BS.S, BS.r, BS.isCall[high_arg])
            res = optimize_ls(BS_high, bvol[high_arg], bdelta[high_arg], x0)
            bvol[high_arg] = self.haganLogNormalApprox(BS_high, res.x[0], res.x[1], res.x[2], res.x[3]) 
        else:
            y_low = np.sort(np.unique(BS.y))[1]
            low_arg = (BS.y <= y_low)
            BS_low = BlackScholes(BS.y[low_arg], BS.expiry, BS.S, BS.r, BS.isCall[low_arg])
            res = optimize_ls(BS_low, bvol[low_arg], bdelta[low_arg], x0)
            bvol[low_arg] = self.haganLogNormalApprox(BS_low, res.x[0], res.x[1], res.x[2], res.x[3])
            y_high = np.sort(np.unique(BS.y))[-2]
            high_arg = (BS.y >= y_high)
            BS_high = BlackScholes(BS.y[high_arg], BS.expiry, BS.S, BS.r, BS.isCall[high_arg])
            res = optimize_ls(BS_high, bvol[high_arg], bdelta[high_arg], x0)
            bvol[high_arg] = self.haganLogNormalApprox(BS_high, res.x[0], res.x[1], res.x[2], res.x[3])
        
        res = optimize_ls(BS, bvol, bdelta, start_point)
        print(res.message)
        
        return res.x
    
    def haganPlot(self, BS, price, param):
        bvol = BS.blackIV(price)
        xline = np.linspace(BS.y.min(), BS.y.max(), 100)
        newBS = copy.deepcopy(BS)
        newBS.y = xline
        plt.figure()
        plt.scatter(BS.y[BS.isCall], bvol[BS.isCall])
        plt.scatter(BS.y[~BS.isCall], bvol[~BS.isCall])
        plt.plot(xline, self.haganLogNormalApprox(newBS, param[0], param[1], param[2], param[3]))
        
        return 0
    
    def computeSABRdelta(self, BS, alpha, beta, nu, rho):
        small_figure = 1e-6
        F_0 = BS.S * math.exp(BS.r * BS.expiry)
        S_plus_h = BS.S + small_figure 
        avg_alpha = (alpha + (rho * nu / math.pow(F_0, beta)) * small_figure)
        vol = self.haganLogNormalApprox(BS, avg_alpha, beta, nu, rho)
        BS_plus_h = BlackScholes(BS.y, BS.expiry, S_plus_h, BS.r, BS.isCall)
        px_f_plus_h = BS_plus_h.black(vol)
        S_minus_h = BS.S - small_figure 
        avg_alpha = (alpha + (rho * nu / math.pow(F_0, beta)) * (-small_figure)) 
        vol = self.haganLogNormalApprox(BS, avg_alpha, beta, nu, rho)
        BS_minus_h = BlackScholes(BS.y, BS.expiry, S_minus_h, BS.r, BS.isCall)
        px_f_minus_h = BS_minus_h.black(vol)
        sabr_delta = computeFirstDerivative(px_f_plus_h, px_f_minus_h, small_figure)
        
        return sabr_delta
    
    def computeSABRvega(self, BS, alpha, beta, nu, rho):
        small_figure = 1e-6
        F_0 = BS.S * math.exp(BS.r * BS.expiry)
        alpha_plus_h = alpha + small_figure
        avg_S = (F_0 + (rho * math.pow(F_0, beta) / nu) * small_figure) * math.exp(-BS.r * BS.expiry)
        BS_plus_h = BlackScholes(BS.y, BS.expiry, avg_S, BS.r, BS.isCall)
        vol = self.haganLogNormalApprox(BS_plus_h, alpha_plus_h , beta, nu, rho)
        px_a_plus_h = BS.black(vol)
        alpha_minus_h = alpha - small_figure 
        avg_S = (F_0 + (rho * math.pow(F_0, beta) / nu) * (-small_figure)) * math.exp(-BS.r * BS.expiry)
        BS_minus_h = BlackScholes(BS.y, BS.expiry, avg_S, BS.r, BS.isCall)
        vol = self.haganLogNormalApprox(BS_minus_h, alpha_minus_h , beta, nu, rho) 
        px_a_minus_h = BS.black(vol)
        sabr_vega = computeFirstDerivative(px_a_plus_h, px_a_minus_h, small_figure)
        return sabr_vega
    
    def computeSABRtheta(self, BS, alpha, beta, nu, rho):
        small_figure = 1e-6
        expiry_plus_h = BS.expiry + small_figure
        BS_plus_h = BlackScholes(BS.y, expiry_plus_h, BS.S, BS.r, BS.isCall)
        vol = self.haganLogNormalApprox(BS_plus_h, alpha, beta, nu, rho)
        px_a_plus_h = BS.black(vol)
        expiry_minus_h = BS.expiry - small_figure
        BS_minus_h = BlackScholes(BS.y, expiry_minus_h, BS.S, BS.r, BS.isCall)
        vol = self.haganLogNormalApprox(BS_minus_h, alpha, beta, nu, rho) 
        px_a_minus_h = BS.black(vol)
        sabr_theta = computeFirstDerivative(px_a_plus_h, px_a_minus_h, small_figure)
        return sabr_theta
    
    def computeSABRgamma(self, BS, alpha, beta, nu, rho):
        small_figure = 1e-6
        F_0 = BS.S * math.exp(BS.r * BS.expiry)
        S_plus_h = BS.S + small_figure 
        avg_alpha = (alpha + (rho * nu / math.pow(F_0, beta)) * small_figure)
        vol = self.haganLogNormalApprox(BS, avg_alpha, beta, nu, rho)
        BS_plus_h = BlackScholes(BS.y, BS.expiry, S_plus_h, BS.r, BS.isCall)
        px_f_plus_h = BS_plus_h.black(vol)
        S_minus_h = BS.S - small_figure 
        avg_alpha = (alpha + (rho * nu / math.pow(F_0, beta)) * (-small_figure)) 
        vol = self.haganLogNormalApprox(BS, avg_alpha, beta, nu, rho)
        BS_minus_h = BlackScholes(BS.y, BS.expiry, S_minus_h, BS.r, BS.isCall)
        px_f_minus_h = BS_minus_h.black(vol)
        vol = self.haganLogNormalApprox(BS, alpha, beta, nu, rho)
        px = BS.black(vol)
        sabr_gamma = computeSecondDerivative(px, px_f_plus_h, px_f_minus_h, small_figure)
        
        return sabr_gamma


