from __future__ import annotations
import pandas as pd, numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def decompose(series, period=6):
    s=pd.Series(series).interpolate().dropna(); return STL(s,period=period,robust=True).fit()

def forecast_rvi(series, steps=3, seasonal_periods=None):
    s=pd.Series(series).dropna().astype(float)
    if len(s)<4: return np.repeat(s.iloc[-1],steps)
    model=ExponentialSmoothing(s,trend='add',seasonal=None if not seasonal_periods else 'add',seasonal_periods=seasonal_periods).fit(optimized=True)
    return model.forecast(steps)
