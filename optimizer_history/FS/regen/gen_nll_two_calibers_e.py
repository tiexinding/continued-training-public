#!/usr/bin/env python3
"""生成 nll_two_calibers_e.json（10-16 恢复，文档 49 L5）。原为一次性命令，时间 2026-10-07T06:30:21.706Z（UTC），
从当时的会话记录逐字取回；此处只加本文件头与工作目录设置（原命令在该目录下运行），其余代码未改动，
个别输入路径的替换见下方 L5 注释。"""
import os as _os
from pathlib import Path as _P
_os.chdir(_P(__file__).resolve().parents[2] / "DATA")
# ---- 以下为原命令
import json,numpy as np,os
rng=np.random.default_rng(0)
def load(p):
    d=json.load(open(p))['nll']; return {k:(np.array(v['s']),np.array(v['c'])) for k,v in d.items()}
paths={'E_vt':'e/diag25/e4/segnll/E_vt.json','E_vfac':'e/diag25/e4/segnll/E_vfac.json','E_vperm':'e/diag25/e4/segnll/E_vperm.json','E_0':'e/diag25/e4/segnll/E_0.json','B_g':'e/diag25/e4/segnll/B_g.json','B_c':'e/diag25/e4/segnll/B_c.json','P_vt':'e/diag25/e4/segnll/P_vt.json','P_0':'e/diag25/e4/segnll/P_0.json',
 'K_vt':'c/diag25/c4/segnll/K_vt.json','K_0':'c/diag25/c4/segnll/K_0.json','K_mvt':'c/diag25/c4/segnll/K_mvt_rec.json','M':'c/diag25/c4/segnll/M.json',
 'R_vt':'d/diag25/d4/segnll/R_vt.json','R_vfac':'d/diag25/d4/segnll/R_vfac.json','R_vperm':'d/diag25/d4/segnll/R_vperm.json','R_vperm_med':'e/diag25/e4/segnll/R_vperm_med.json' if os.path.exists('e/diag25/e4/segnll/R_vperm_med.json') else None}
D={k:load(p) for k,p in paths.items() if p and os.path.exists(p)}; print('有逐题文件:',list(D))
def both(a,b,key):
    sa,ca=D[a][key]; sb,cb=D[b][key]
    item=(sa/ca-sb/cb); tok=(sa.sum()/ca.sum()-sb.sum()/cb.sum())
    n=len(item); bs=[]; bt=[]
    for _ in range(2000):
        i=rng.integers(0,n,n); bs.append(item[i].mean()); bt.append(sa[i].sum()/ca[i].sum()-sb[i].sum()/cb[i].sum())
    return item.mean(),np.percentile(bs,[2.5,97.5]),tok,np.percentile(bt,[2.5,97.5])
out={}
for a,b,key in [('E_vfac','E_vt','probe_bio' if 'probe_bio' in D.get('E_vt',{}) else 'probe_gsm8k'),('E_vperm','E_vt','probe_gsm8k'),('E_0','E_vt','probe_gsm8k'),('R_vfac','R_vt','probe_gsm8k'),('R_vperm','R_vt','probe_gsm8k'),('R_vperm_med','R_vt','probe_gsm8k'),('P_vt','K_vt','probe_mbpp'),('P_0','K_0','probe_mbpp'),('P_vt','P_0','probe_mbpp'),('P_vt','M','probe_mbpp'),('K_vt','M','probe_mbpp'),('K_mvt','K_vt','probe_mbpp'),('P_vt','K_vt','probe_gsm8k'),('E_vfac','E_vt','probe_gsm8k')]:
    if a not in D or b not in D or key not in D[a]: print('缺',a,b,key); continue
    im,ic,tm,tc=both(a,b,key); out[f'{a}-{b}|{key}']=dict(item=im,item_ci=ic.tolist(),token=tm,token_ci=tc.tolist())
    print(f'{a:12}-{b:6} {key:12} 逐题等权 {im:+.4f} [{ic[0]:+.4f},{ic[1]:+.4f}]  token加权 {tm:+.4f} [{tc[0]:+.4f},{tc[1]:+.4f}]')
print('E_vt 键:',list(D['E_vt']))
json.dump(out,open('e/nll_two_calibers_e.json','w'),indent=1)
