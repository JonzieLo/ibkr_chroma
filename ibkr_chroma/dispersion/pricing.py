"""
Options Pricing Models

- BSMDividendModel: Black-Scholes-Merton with continuous dividend yield q (European Index Options).
- AmericanBinomialModel: CRR Tree with early exercise boundaries (American Single-Stock & Index Options).
"""

import math
from typing import Tuple
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


class BSMDividendModel:
    """Black-Scholes-Merton with continuous dividend yield."""
    
    @staticmethod
    def price(
        spot: float,
        strike: float,
        tau: float,
        vol: float,
        rate: float,
        div_yield: float,
        is_call: bool = True
    ) -> float:
        if tau <= 0.0 or vol <= 0.0:
            return max(0.0, (spot - strike) if is_call else (strike - spot))

        df_r = math.exp(-rate * tau)
        df_q = math.exp(-div_yield * tau)
        std_dev = vol * math.sqrt(tau)

        d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * vol * vol) * tau) / std_dev
        d2 = d1 - std_dev

        if is_call:
            return spot * df_q * norm.cdf(d1) - strike * df_r * norm.cdf(d2)
        return strike * df_r * norm.cdf(-d2) - spot * df_q * norm.cdf(-d1)

    @staticmethod
    def greeks(spot: float, strike: float, tau: float, vol: float, rate: float, div_yield: float, is_call: bool = True) -> dict:
        if tau <= 0.0 or vol <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

        df_r = math.exp(-rate * tau)
        df_q = math.exp(-div_yield * tau)
        sqrt_tau = math.sqrt(tau)
        std_dev = vol * sqrt_tau

        d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * vol * vol) * tau) / std_dev
        d2 = d1 - std_dev

        delta = df_q * norm.cdf(d1) if is_call else df_q * (norm.cdf(d1) - 1.0)
        gamma = df_q * norm.pdf(d1) / (spot * std_dev)
        vega = spot * df_q * norm.pdf(d1) * sqrt_tau

        theta_term = -(spot * df_q * norm.pdf(d1) * vol) / (2.0 * sqrt_tau)
        if is_call:
            theta = theta_term - rate * strike * df_r * norm.cdf(d2) + div_yield * spot * df_q * norm.cdf(d1)
        else:
            theta = theta_term + rate * strike * df_r * norm.cdf(-d2) - div_yield * spot * df_q * norm.cdf(-d1)

        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


class BinomialModel:
    """Cox-Ross-Rubinstein model by Binomial Tree (150 steps)."""
    
    def __init__(self, steps: int = 150):
        self.steps = steps

    def price(self, spot: float, strike: float, tau: float, vol: float, rate: float, div_yield: float, is_call: bool = True) -> float:
        if tau <= 0.0 or vol <= 0.0:
            return max(0.0, (spot - strike) if is_call else (strike - spot))

        dt = tau / self.steps
        u = math.exp(vol * math.sqrt(dt))
        d = 1.0 / u
        p = (math.exp((rate - div_yield) * dt) - d) / (u - d)
        disc = math.exp(-rate * dt)
        sign = 1.0 if is_call else -1.0

        # Terminal payoffs
        j = np.arange(self.steps + 1)
        st = spot * (u ** (self.steps - j)) * (d ** j)
        values = np.maximum(sign * (st - strike), 0.0)

        # Backward induction checking early exercise boundary at each node
        for step in range(self.steps - 1, -1, -1):
            j_step = np.arange(step + 1)
            st_node = spot * (u ** (step - j_step)) * (d ** j_step)
            hold_value = disc * (p * values[:-1] + (1.0 - p) * values[1:])
            intrinsic = np.maximum(sign * (st_node - strike), 0.0)
            values = np.maximum(hold_value, intrinsic)

        return float(values[0])

    def greeks(self, spot: float, strike: float, tau: float, vol: float, rate: float, div_yield: float, is_call: bool = True) -> dict:
        """Computes finite-difference Greeks for American options."""
        h_s = max(0.01, spot * 0.005)
        h_v = 0.005

        p_base = self.price(spot, strike, tau, vol, rate, div_yield, is_call)
        p_up = self.price(spot + h_s, strike, tau, vol, rate, div_yield, is_call)
        p_down = self.price(spot - h_s, strike, tau, vol, rate, div_yield, is_call)

        delta = (p_up - p_down) / (2.0 * h_s)
        gamma = (p_up - 2.0 * p_base + p_down) / (h_s ** 2)

        p_v_up = self.price(spot, strike, tau, vol + h_v, rate, div_yield, is_call)
        p_v_down = self.price(spot, strike, tau, vol - h_v, rate, div_yield, is_call)
        vega = (p_v_up - p_v_down) / (2.0 * h_v)

        dt = 1.0 / 365.0
        p_t = self.price(spot, strike, max(1e-4, tau - dt), vol, rate, div_yield, is_call)
        theta = (p_t - p_base) / dt

        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}