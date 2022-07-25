# -*- coding: utf-8 -*-
"""
Created on 2022/7/15 11:23
@author: jhyu
"""
import numpy as np
import scipy.stats as spst

from . import option_abc as opt


class Bsm(opt.OptABC):
    """
    Black-Scholes-Merton (BSM) model for option pricing.

    Underlying price is assumed to follow a geometric Brownian motion.
    """

    @staticmethod
    def price_formula(
        strike, spot, sigma, texp, cp=1, intr=0.0, divr=0.0, is_fwd=False
    ):
        """
        Black-Scholes-Merton model call/put option pricing formula (static method)

        Args:
            strike: strike price
            spot: spot (or forward)
            sigma: model volatility
            texp: time to expiry
            cp: 1/-1 for call/put option
            sigma: model volatility
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.

        Returns:
            Vanilla option price
        """
        disc_fac = np.exp(-texp * intr)
        fwd = np.array(spot) * (1.0 if is_fwd else np.exp(-texp * divr) / disc_fac)

        sigma_std = np.maximum(np.array(sigma) * np.sqrt(texp), np.finfo(float).eps)

        # don't directly compute d1 just in case sigma_std is infty
        d1 = np.log(fwd / strike) / sigma_std
        d2 = d1 - 0.5 * sigma_std
        d1 += 0.5 * sigma_std

        cp = np.array(cp)
        price = fwd * spst.norm.cdf(cp * d1) - strike * spst.norm.cdf(cp * d2)
        price *= cp * disc_fac
        return price

    def price(self, strike, spot, texp, cp=1):
        return self.price_formula(
            strike=strike, spot=spot, texp=texp, cp=cp, **self.params_kw()
        )

    def delta(self, strike, spot, texp, cp=1):

        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), np.finfo(float).eps)
        d1 = np.log(fwd / strike) / sigma_std + 0.5 * sigma_std

        delta = cp * spst.norm.cdf(cp * d1)  # formula according to wikipedia
        delta *= df if self.is_fwd else divf
        return delta

    def cdf(self, strike, spot, texp, cp=1):

        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), np.finfo(float).eps)
        d2 = np.log(fwd / strike) / sigma_std - 0.5 * sigma_std
        cdf = spst.norm.cdf(cp * d2)  # formula according to wikipedia
        return cdf

    def gamma(self, strike, spot, texp, cp=1):

        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), 100 * np.finfo(float).eps)
        d1 = np.log(fwd / strike) / sigma_std + 0.5 * sigma_std

        gamma = (
            df * spst.norm.pdf(d1) / fwd / sigma_std
        )  # formula according to wikipedia
        if not self.is_fwd:
            gamma *= (divf / df) ** 2
        return gamma

    def theta(self, strike, spot, texp, cp=1):

        fwd, df, divf = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), 100 * np.finfo(float).eps)
        d1 = np.log(fwd / strike) / sigma_std
        d2 = d1 - 0.5 * sigma_std
        d1 += 0.5 * sigma_std

        # still not perfect; need to consider the derivative w.r.t. divr and is_fwd = True
        theta = -0.5 * spst.norm.pdf(d1) * fwd * self.sigma / np.sqrt(
            texp
        ) - cp * self.intr * strike * spst.norm.cdf(cp * d2)
        theta *= df
        return theta

    def vega(self, strike, spot, texp, cp=1):

        fwd, df, _ = self._fwd_factor(spot, texp)

        sigma_std = np.maximum(self.sigma * np.sqrt(texp), np.finfo(float).eps)
        d1 = np.log(fwd / strike) / sigma_std + 0.5 * sigma_std

        vega = (
            df * fwd * spst.norm.pdf(d1) * np.sqrt(texp)
        )  # formula according to wikipedia
        return vega

