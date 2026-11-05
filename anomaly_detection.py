from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

RISK_ORDER=['Normal','Watch','Moderate Risk','High Risk','Critical']

def engineer_features(df):
    x=df.sort_values(['block_id','date']).copy()
    x['difference']=x.observed_rvi-x.expected_rvi
    x['percent_deviation']=100*x.difference/x.expected_rvi.replace(0,np.nan)
    x['z_score']=x.difference/x.expected_std.clip(lower=0.03)
    x['growth_rate']=x.groupby('block_id').observed_rvi.diff()/x.groupby('block_id').crop_age_days.diff().replace(0,np.nan)
    x['rolling_mean']=x.groupby('block_id').observed_rvi.transform(lambda s:s.rolling(3,min_periods=1).mean())
    x['trend_delta']=x.observed_rvi-x.rolling_mean
    return x

def classify_risk(percent):
    return pd.cut(percent,[-np.inf,-40,-30,-20,-10,np.inf],labels=['Critical','High Risk','Moderate Risk','Watch','Normal']).astype(str)

def score_rules(df):
    x=engineer_features(df); x['risk_level']=classify_risk(x.percent_deviation)
    x['rule_score']=np.clip(-x.percent_deviation/40,0,1); return x

def isolation_forest(df, contamination=0.08, random_state=42):
    x=df.copy(); cols=['observed_rvi','percent_deviation','z_score','growth_rate','trend_delta']
    X=x[cols].replace([np.inf,-np.inf],np.nan).fillna(0)
    m=IsolationForest(contamination=contamination,random_state=random_state).fit(X)
    x['if_score']=-m.decision_function(X); x['if_anomaly']=(m.predict(X)==-1).astype(int); return x,m

def train_random_forest(df, label='is_problem', random_state=42):
    cols=['observed_rvi','expected_rvi','percent_deviation','z_score','growth_rate','crop_age_days']
    X=df[cols].replace([np.inf,-np.inf],np.nan).fillna(0); y=df[label].astype(int)
    m=RandomForestClassifier(n_estimators=400,class_weight='balanced',random_state=random_state,n_jobs=-1).fit(X,y)
    return m,cols

def classification_metrics(y,p,prob=None):
    pr,re,f1,_=precision_recall_fscore_support(y,p,average='binary',zero_division=0)
    out={'precision':pr,'recall':re,'f1':f1}
    if prob is not None and len(np.unique(y))==2: out['roc_auc']=roc_auc_score(y,prob)
    return out
