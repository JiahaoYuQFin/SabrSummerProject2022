# -*- coding: utf-8 -*-
"""
Created on 2022/7/22 15:30
@author: jhyu
"""
import scipy.optimize as sopt
import numpy as np
import copy
import abc


class OptABC(abc.ABC):
    sigma, texp, intr, divr = None, None, 0.0, 0.0
    is_fwd = False

    IMPVOL_TOL = 1e-10
    IMPVOL_MAXVOL = 999.99

    def __init__(self, sigma, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatility
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        self.sigma = sigma
        self.intr = intr
        self.divr = divr
        self.is_fwd = is_fwd

    def params_kw(self):
        """
        Model parameters in dictionary
        """
        params = {
            "sigma": self.sigma,
            "intr": self.intr,
            "divr": self.divr,
            "is_fwd": self.is_fwd,
        }
        return params

    def _fwd_factor(self, spot, texp):
        """
        Forward, discount factor, dividend factor

        Args:
            spot: spot (or forward) price
            texp: time to expiry

        Returns:
            (forward, discounting factor, dividend factor)
        """
        df = np.exp(-self.intr * np.array(texp))
        if self.is_fwd:
            divf = 1
            fwd = np.array(spot)
        else:
            divf = np.exp(-self.divr * np.array(texp))
            fwd = np.array(spot) * divf / df
        return fwd, df, divf

    @abc.abstractmethod
    def price(self, strike, spot, texp, cp=1):
        """
        Call/put option price.

        Args:
            strike: strike price.
            spot: spot (or forward) price.
            texp: time to expiry.
            cp: 1/-1 for call/put option.

        Returns:
            option price
        """
        return NotImplementedError

    def impvol_brentq(self, price, strike, spot, texp, cp=1, setval=False):
        """
        Implied volatility using Brent's method. Slow but robust implementation.

        Args:
            price: option price
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            cp: 1/-1 for call/put
            setval: if True, sigma is set with the solved implied volatility

        Returns:
            implied volatility
        """
        fwd, df, _ = self._fwd_factor(spot, texp)

        kk = strike / fwd  # strike / fwd
        price_std = price / df / fwd  # forward price / fwd

        model = copy.copy(self)
        model.sigma = 1e-64
        p_min = model.price(kk, 1, texp, cp)
        model.sigma = self.IMPVOL_MAXVOL
        p_max = model.price(kk, 1, texp, cp)

        scalar_output = np.isscalar(price) & np.isscalar(p_min)
        ones_like = np.ones_like(np.atleast_1d(price + p_min))

        sigma = np.empty(ones_like.shape).flatten()
        sigma.fill(np.nan)
        price_flat = (ones_like * price_std).flatten()
        p_min = (ones_like * p_min).flatten()
        p_max = (ones_like * p_max).flatten()
        texp_flat = (ones_like * texp).flatten()
        kk_flat = (ones_like * kk).flatten()
        cp_flat = (ones_like * cp).flatten()

        def iv_func(_sigma):
            model.sigma = _sigma
            return model.price(_strike, 1.0, _texp, _cp) - _price

        for k in range(len(sigma)):
            _cp = cp_flat[k]
            _texp = texp_flat[k]
            _strike = kk_flat[k]
            _price = price_flat[k]

            if np.abs(_price - p_min[k]) < self.IMPVOL_TOL:
                sigma[k] = 0.0
            elif np.abs(_price - p_max[k]) < self.IMPVOL_TOL:
                sigma[k] = self.IMPVOL_MAXVOL
            elif _price < p_min[k] or p_max[k] < _price:
                sigma[k] = np.nan
            else:
                sigma[k] = sopt.brentq(iv_func, 0.0, 10)

        if scalar_output:
            sigma = sigma[0]
        else:
            sigma = sigma.reshape(ones_like.shape)
        if setval:
            self.sigma = sigma
        return sigma

    impvol = impvol_brentq

    def delta_numeric(self, strike, spot, texp, cp=1):
        """
        Option model delta (sensitivity to price) by finite difference

        Args:
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns:
            delta value
        """
        h = spot * 1e-6
        delta = (
            self.price(strike, spot + h, texp, cp) -
            self.price(strike, spot - h, texp, cp)
        ) / (2 * h)
        return delta

    def gamma_numeric(self, strike, spot, texp, cp=1):
        """
        Option model gamma (2nd derivative to price) by finite difference

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put

        Returns:
            Delta with numerical derivative
        """
        h = spot * 1e-6
        gamma = (
            self.price(strike, spot + h, texp, cp)
            - 2 * self.price(strike, spot, texp, cp)
            + self.price(strike, spot - h, texp, cp)
        ) / (h * h)
        return gamma

    def vega_numeric(self, strike, spot, texp, cp=1):
        """
        Option model vega (sensitivity to volatility) by finite difference

        Args:
            strike: strike price
            spot: spot (or forward) price
            texp: time to expiry
            cp: 1/-1 for call/put option

        Returns:
            vega value
        """
        h = 1e-6
        model = copy.copy(self)
        model.sigma += h
        p_up = model.price(strike, spot, texp, cp)
        model.sigma -= 2 * h
        p_dn = model.price(strike, spot, texp, cp)

        vega = (p_up - p_dn) / (2 * h)
        return vega

    def theta_numeric(self, strike, spot, texp, cp=1):
        """
        Option model thegta (sensitivity to time-to-maturity) by finite difference

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp: 1/-1 for call/put

        Returns:
            theta value
        """
        dt = np.minimum(5/60/24/365.25, texp)  # 5 minutes
        theta = self.price(strike, spot, texp - dt, cp) - self.price(
            strike, spot, texp, cp
        )
        theta /= dt
        return theta

    # create aliases
    delta = delta_numeric
    gamma = gamma_numeric
    vega = vega_numeric
    theta = theta_numeric
