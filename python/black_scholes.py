import numpy as np
from scipy.stats import norm

def black_scholes(s, k, t, r, sigma, option_type = "call"):
    """_summary_

    Args:
        s int: Current Stock Price
        k int: Strike Price
        t int: Time to Maturity
        r (_type_): Risk-free Rate
        sigma (_type_): Volatility of Variance
        option_type (str, optional):"call" or "put - Defaults to "call".

    Returns:
        _type_: _description_
    """
    
    d1 = (np.log(s/k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    if option_type == "call":
        price = s * norm.cdf(d1) - k * np.exp(-r, t) * norm.cdf(d2)
    elif option_type == "put":
        price = k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
    
    return price