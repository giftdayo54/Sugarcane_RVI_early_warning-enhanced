from __future__ import annotations
def build_alerts(report):
    order={'Normal':0,'Watch':1,'Moderate Risk':2,'High Risk':3,'Critical':4}; alerts=[]
    for _,r in report.iterrows():
        if order.get(r.risk_level,0)>=3:
            alerts.append(f"Block {r.block_id} has RVI {abs(r.percent_deviation):.0f}% below expected growth and is classified as {r.risk_level}.")
    return alerts
