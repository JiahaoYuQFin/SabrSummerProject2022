# -*- coding: utf-8 -*-
"""
Created on 2022/7/15 10:45
@author: jhyu
"""
from . import option_abc as opt
import scipy.optimize as sopt
from . import bsm
import numpy as np
import copy
import abc


class SabrABC(opt.OptABC):
    vov, beta, rho = 0.0, 1.0, 0.0
    model_type = "SABR"
    _base_beta = None
    approx_order = 1  # order in texp: 0: leading order, 1: first order

    def __init__(
        self, sigma, vov=0.0, rho=0.0, beta=1.0, intr=0.0, divr=0.0, is_fwd=False
    ):
        """
        Args:
            sigma: model volatility at t=0
            vov: volatility of volatility
            rho: correlation between price and volatility
            beta: elasticity parameter. 1.0 by default
            intr: interest rate (domestic interest rate)
            divr: dividend/convenience yield (foreign interest rate)
            is_fwd: if True, treat `spot` as forward price. False by default.
        """
        super().__init__(sigma, intr=intr, divr=divr, is_fwd=is_fwd)
        self.vov = vov
        self.rho = rho
        self.beta = beta

    def params_kw(self):
        params = super().params_kw()
        extra = {"vov": self.vov, "beta": self.beta, "rho": self.rho}
        return {**params, **extra}  # Py 3.9, params | extra

    def _variables(self, fwd, texp):
        betac = 1.0 - self.beta
        alpha = self.sigma / np.power(fwd, betac)  # if self.beta > 0.0 else self.sigma
        rho2 = self.rho * self.rho
        rhoc = np.sqrt(1.0 - rho2)
        vovn = self.vov * np.sqrt(np.maximum(texp, 1e-64))
        return alpha, betac, rhoc, rho2, vovn

    def _m_base(self, vol, is_fwd=None):
        """
        Create base model based on _base_beta value: `Norm` for 0, Cev for (0,1), and `Bsm` for 1
        If `_base_beta` is None, use `base` instead.

        Args:
            vol: base model volatility

        Returns: model
        """
        base_beta = self._base_beta or self.beta
        if is_fwd is None:
            is_fwd = self.is_fwd
        if np.isclose(base_beta, 1):
            return bsm.Bsm(vol, intr=self.intr, divr=self.divr, is_fwd=is_fwd)
        else:
            raise ValueError

    @abc.abstractmethod
    def vol_for_price(self, strike, spot, texp):
        """
        Equivalent volatility of the SABR model

        Args:
            strike: strike price
            spot: spot (or forward)
            texp: time to expiry

        Returns:
            equivalent volatility
        """
        return NotImplementedError

    @staticmethod
    def _vv(zz, rho):
        return np.sqrt(1 + zz * (zz + 2 * rho))

    @staticmethod
    def _hh(zz, rho):
        """
        H(z) in the paper

        Args:
            zz: array
            rho: scalar

        Returns: H(z)

        """
        rho2 = rho * rho
        # initalization with expansion for for small |zz|
        xx_zz = 1 - (zz / 2) * (
                rho - zz * (rho2 - 1 / 3 - (5 * rho2 - 3) / 4 * rho * zz)
        )

        yy = SabrABC._vv(zz, rho)
        eps = 1e-5

        with np.errstate(divide="ignore", invalid="ignore"):  # suppress error for zz=0
            # replace negative zz
            xx_zz = np.where(
                zz > -eps, xx_zz, np.log((1 - rho) / (yy - (zz + rho))) / zz
            )
            # replace positive zz
            xx_zz = np.where(
                zz < eps, xx_zz, np.log((yy + (zz + rho)) / (1 + rho)) / zz
            )

        return 1.0 / xx_zz

    def price(self, strike, spot, texp, cp=1):
        vol = self.vol_for_price(strike, spot, texp)
        m_vol = self._m_base(vol)
        price = m_vol.price(strike, spot, texp, cp=cp)
        return price


class SabrHagan2002(SabrABC):
    """
    SABR model with Hagan's implied volatility approximation for 0<beta<=1.

    References:
        Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). Managing Smile Risk.
        Wilmott, September, 84–108.
    """

    _base_beta = 1.0  # should not be changed

    def vol_for_price(self, strike, spot, texp):
        # fwd, spot, sigma may be either scalar or np.array.
        # texp, vov, rho, beta should be scholar values

        if texp <= 0.0:
            return 0.0
        fwd, _, _ = self._fwd_factor(spot, texp)
        alpha, betac, rhoc, rho2, vovn = self._variables(spot, texp)
        betac2 = betac ** 2

        log_kk = np.log(fwd / strike)
        log_kk2 = log_kk * log_kk
        pow_kk = np.power(strike / fwd, betac / 2)

        pre1 = pow_kk * (1 + betac2 / 24 * log_kk2 * (1 + betac2 / 80 * log_kk2))

        term02 = (2 - 3 * rho2) / 24 * self.vov ** 2
        term11 = alpha * self.vov * self.rho * self.beta / 4 / pow_kk
        term20 = (betac * alpha / pow_kk) ** 2 / 24

        if self.approx_order == 0:
            vol = 1.0
        else:
            vol = 1.0 + texp * (term02 + term11 + term20)

        zz = pow_kk * log_kk * self.vov / np.maximum(alpha, np.finfo(float).eps)
        hh = self._hh(
            -zz, self.rho
        )  # note we pass -zz becaues hh(zz) definition is different
        vol *= alpha * hh / pre1  # bsm vol
        return vol

    def calibrate3(
        self, price_or_vol3, strike3, spot, texp, cp=1, setval=False, is_vol=True
    ):
        """
        Given option prices or implied vols at 3 strikes, compute the sigma, vov, rho to fit the data using `scipy.optimize.root`.
        If prices are given (is_vol=False) convert the prices to vol first.

        Args:
            price_or_vol3: 3 prices or 3 volatilities (depending on `is_vol`)
            strike3: 3 strike prices
            spot: spot price
            texp: time to expiry
            cp: cp
            setval: if True, set sigma, vov, rho values
            is_vol: if True, `price_or_vol3` are volatilities.

        Returns:
            Dictionary of `sigma`, `vov`, and `rho`.
        """
        model = copy.copy(self)

        if is_vol:
            vol3 = price_or_vol3
        else:
            vol3 = self._m_base(None).impvol(price_or_vol3, strike3, spot, texp, cp=cp)
        
        def iv_func(x):
            model.sigma = np.exp(x[0])
            model.vov = np.exp(x[1])
            model.rho = np.tanh(x[2])
            err = model.vol_for_price(strike3, spot, texp=texp) - vol3
            return err

        sol = sopt.root(iv_func, np.array([np.log(vol3[1]), -1, 0.0]))
        params = {
            "sigma": np.exp(sol.x[0]),
            "vov": np.exp(sol.x[1]),
            "rho": np.tanh(sol.x[2]),
        }

        if setval:
            self.sigma, self.vov, self.rho = (
                params["sigma"],
                params["vov"],
                params["rho"],
            )

        return params
