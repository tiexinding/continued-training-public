#!/usr/bin/env python3
"""生成 humaneval_pairs_e_rescore.json（10-16 恢复，文档 49 L5）。原为一次性命令，时间 2026-10-07T07:03:26.363Z（UTC），
从当时的会话记录逐字取回；此处只加本文件头与工作目录设置（原命令在该目录下运行），其余代码未改动，
个别输入路径的替换见下方 L5 注释。"""
import os as _os
from pathlib import Path as _P
_os.chdir(_P(__file__).resolve().parents[2])
# ---- 以下为原命令
import json,numpy as np,math
R=json.load(open('DATA/e/humaneval_e_rescore.json'))
def mcn(a,b):
    a=np.array(a,bool); b=np.array(b,bool); g=int(np.sum(a&~b)); l=int(np.sum(~a&b)); n=g+l; p=min(1.0,2*sum(math.comb(n,i) for i in range(0,min(g,l)+1))/2**n) if n else 1.0; return g,l,p
pairs=[('E_vt','E_0'),('E_vfac','E_vt'),('E_vperm','E_vt'),('E_vfac','E_0'),('E_vperm','E_0'),('E_vt','B_g'),('E_0','B_g'),('B_c','B_g'),('P_vt','P_0'),('R_vperm_med','B_g')]
out={}
for rule in ('v2','v3_autoindent'):
    print('规则',rule); res={}
    for a,b in pairs:
        g,l,p=mcn(R[a][rule]['passed'],R[b][rule]['passed']); res[f'{a}|{b}']=dict(gain=g,loss=l,p=p,acc_a=R[a][rule]['pass_rate'],acc_b=R[b][rule]['pass_rate'])
    ps=sorted([(v['p'],k) for k,v in res.items()]); m=len(ps)
    for i,(p,k) in enumerate(ps): res[k]['p_holm']=min(1.0,max(ps[j][0]*(m-j) for j in range(i+1)))
    for k,v in res.items(): print(f"  {k:20} {v['acc_a']:.3f} 对 {v['acc_b']:.3f} 得/失 {v['gain']}/{v['loss']} p={v['p']:.4f} Holm={v['p_holm']:.3f}")
    out[rule]=res
# 无缩进函数体输出率（格式指标）
H=json.load(open('DATA/e/diag25/e4/humaneval_e.json'))
import re
def bodyonly(o,ep):
    m=re.search(r"```(?:python)?\n(.*?)```",o,re.S); b=m.group(1) if m else o
    if ('def '+ep) in b: return 'full'
    first=next((l for l in b.split("\n") if l.strip()),"")
    return 'body_unindented' if first and not first.startswith((' ','\t')) else 'body_indented'
# L5：原读当时会话临时目录中的 humaneval.jsonl（已不存在）；改读同一题集的 parquet（164 题，官方顺序）
import pandas as _pd
rows=[dict(entry_point=r.entry_point) for r in _pd.read_parquet('DATA/h/humaneval.parquet').itertuples()]
import collections
fmt={}
for arm in H:
    c=collections.Counter(bodyonly(o,r['entry_point']) for o,r in zip(H[arm]['outs'],rows)); fmt[arm]={k:v/164 for k,v in c.items()}
    print(arm,{k:round(v,2) for k,v in fmt[arm].items()})
out['format']=fmt
json.dump(out,open('DATA/e/humaneval_pairs_e_rescore.json','w'),indent=1)
