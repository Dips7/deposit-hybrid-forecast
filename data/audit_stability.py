"""Reproduce development diagnostics without fitting models or calculating new p-values."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
run=ROOT/'outputs/development_fit_v1'
sources=[run/'evaluation_DEVELOPMENT_FIT_V1.csv',run/'inference_summary_DEVELOPMENT_FIT_V1.json',ROOT/'outputs/development/metrics_HISTORICAL_REPLAY_V1.csv']
d=pd.read_csv(sources[0]);d=d[d.is_eligible_for_evaluation==True].copy()
d['origin_year']=d.origin_period.str[:4];d['target_year']=d.target_period.str[:4]
annual=[]
for basis in ['origin_year','target_year']:
 for (year,proc),g in d.groupby([basis,'procedure_id']):
  annual.append(dict(year_basis=basis,year=year,procedure=proc,n=len(g),mae=g.abs_error.mean(),rmse=np.sqrt(g.sq_error.mean())))
pd.DataFrame(annual).to_csv(OUT/'YEARLY_METRICS.csv',index=False)
tails=[]
for proc,g in d.groupby('procedure_id'):
 rec=dict(procedure=proc,n=len(g),max_abs_error=g.abs_error.max())
 for q in [.5,.75,.9,.95,.99]:rec[f'p{int(q*100)}']=g.abs_error.quantile(q)
 for k in [9,10]:rec[f'top_{k}_squared_loss_share']=g.sq_error.nlargest(k).sum()/g.sq_error.sum()
 tails.append(rec)
pd.DataFrame(tails).to_csv(OUT/'TAIL_METRICS.csv',index=False)
piv=d.pivot(index='origin_period',columns='procedure_id',values='actual_error')
piv.corr().to_csv(OUT/'ERROR_CORRELATION.csv')
abs_p=piv.abs();annual_p=d.groupby(['origin_year','procedure_id']).abs_error.mean().unstack()
pd.DataFrame([dict(reference=ref,origin_wins=int((abs_p.H_OMO<abs_p[ref]).sum()),origins=len(abs_p),annual_wins=int((annual_p.H_OMO<annual_p[ref]).sum()),years=len(annual_p)) for ref in ['E_S','E_X','E_EQ','H_Y']]).to_csv(OUT/'WIN_COUNTS.csv',index=False)
paired=d[d.procedure_id=='H_OMO'][['origin_period','target_period','y_actual','actual_error','sq_error']].merge(d[d.procedure_id=='H_Y'][['origin_period','actual_error','sq_error']],on='origin_period',suffixes=('_H_OMO','_H_Y'))
paired['sq_loss_H_OMO_minus_H_Y']=paired.sq_error_H_OMO-paired.sq_error_H_Y
paired.to_csv(OUT/'OMO_INCREMENT_ORIGIN_LOSSES.csv',index=False)
inf=json.loads(sources[1].read_text())
pd.DataFrame([dict(block_family=key,comparison=c['comparison_id'],reference=c['reference'],holm_p=c['holm_adjusted_p'],raw_p=c['raw_p_value']) for key,v in inf.items() if isinstance(v,dict) and 'comparisons' in v for c in v['comparisons']]).to_csv(OUT/'SAVED_INFERENCE_VALUES.csv',index=False)
replay=pd.read_csv(sources[2]);replay.to_csv(OUT/'AUTHENTICATED_REPLAY_METRICS.csv',index=False)
summary=dict(origin_start=d.origin_period.min(),origin_end=d.origin_period.max(),n_origins=d.origin_period.nunique(),n_rows=len(d),total_sq_loss_increment=paired.sq_loss_H_OMO_minus_H_Y.sum(),mid_2020_four_origin_sq_loss_increment=paired.loc[paired.origin_period.between('2020-06','2020-09'),'sq_loss_H_OMO_minus_H_Y'].sum(),no_new_fits=True,no_new_p_values=True,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(OUT/'STABILITY_AUDIT_SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
