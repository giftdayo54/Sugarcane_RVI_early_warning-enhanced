from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def add_crop_age(df):
    x=df.copy(); x['date']=pd.to_datetime(x.date); x['planting_date']=pd.to_datetime(x.planting_date)
    x['crop_age_days']=(x.date-x.planting_date).dt.days.clip(lower=0); return x

def fit_healthy_baseline(df, group_cols=('variety','crop_type'), age_bin_days=14, min_samples=5):
    x=add_crop_age(df); x=x[x['healthy'].fillna(False)].copy()
    x['age_bin']=(x.crop_age_days//age_bin_days)*age_bin_days
    keys=list(group_cols)+['age_bin']
    b=x.groupby(keys,dropna=False).observed_rvi.agg(expected_rvi='mean',expected_std='std',n='size').reset_index()
    b=b[b.n>=min_samples]; b['expected_std']=b.expected_std.fillna(0.05).clip(lower=0.03)
    return b.sort_values(keys)

def attach_baseline(df, baseline, group_cols=('variety','crop_type'), age_bin_days=14):
    x=add_crop_age(df); x['age_bin']=(x.crop_age_days//age_bin_days)*age_bin_days
    return x.merge(baseline,on=[*group_cols,'age_bin'],how='left')

def regression_metrics(y, pred):
    ok=np.isfinite(y)&np.isfinite(pred); y=np.asarray(y)[ok]; pred=np.asarray(pred)[ok]
    return {'RMSE':mean_squared_error(y,pred)**0.5,'MAE':mean_absolute_error(y,pred),'R2':r2_score(y,pred)}
