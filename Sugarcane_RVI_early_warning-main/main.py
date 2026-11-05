from pathlib import Path
import pandas as pd
from src.utils import load_config,ensure_dirs,get_logger
from src.baseline_model import fit_healthy_baseline,attach_baseline
from src.anomaly_detection import score_rules,isolation_forest
from src.reporting import block_report,export_vectors
from src.alerts import build_alerts

def run(config_path='config/config.yaml'):
    cfg=load_config(config_path); log=get_logger(); out=Path(cfg['paths']['outputs']); ensure_dirs(out/'reports',out/'vectors',out/'rasters')
    obs=pd.read_csv(cfg['paths']['observations'])
    base=fit_healthy_baseline(obs,age_bin_days=cfg['baseline']['age_bin_days'],min_samples=cfg['baseline']['min_samples'])
    base.to_csv(out/'reports/healthy_baseline.csv',index=False)
    x=attach_baseline(obs,base,age_bin_days=cfg['baseline']['age_bin_days']); x=score_rules(x); x,model=isolation_forest(x)
    report=block_report(x); report.to_csv(out/'reports/block_report.csv',index=False)
    export_vectors(cfg['paths']['blocks'],report,out/'vectors/block_status')
    (out/'reports/alerts.txt').write_text('\n'.join(build_alerts(report)),encoding='utf-8')
    log.info('Processed %d observations and %d blocks',len(x),len(report)); return report
if __name__=='__main__': run()
