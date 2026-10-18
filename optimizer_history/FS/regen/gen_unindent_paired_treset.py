#!/usr/bin/env python3
"""生成 unindent_paired_treset.json（10-16 恢复，文档 49 L5）。原为一次性命令，时间 2026-10-07T15:14:22.358Z（UTC），
从当时的会话记录逐字取回；此处只加本文件头与工作目录设置（原命令在该目录下运行），其余代码未改动，
个别输入路径的替换见下方 L5 注释。"""
import os as _os
from pathlib import Path as _P
_os.chdir(_P(__file__).resolve().parents[2] / "DATA")
# ---- 以下为原命令
import json,numpy as np,math
H7=json.load(open('f/diag25/f1/heval_f7.json')); H14=json.load(open('f/diag25/f1/heval_f14.json'))
U={}
for H in (H7,H14):
    for k,v in H.items(): U[k]=np.array([c=='unind' for c in v['cats']])
def mcn(a,b):
    g=int((a&~b).sum());l=int((~a&b).sum());n=g+l;return g,l,(min(1,2*sum(math.comb(n,i) for i in range(min(g,l)+1))/2**n) if n else 1)
rng=np.random.default_rng(0)
def ci(a,b):
    d=a.astype(float)-b; bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)]; return d.mean(),np.percentile(bs,[2.5,97.5])
pairs=[('E_v','E_vt'),('E_vfac_t0','E_vfac'),('E_vperm_t0','E_vperm'),('E_v','E_0'),('E_vfac_t0','E_v'),('E_vperm_t0','E_v'),('E_vperm_t0','E_vfac_t0')]
out={}
for a,b in pairs:
    if a in U and b in U:
        g,l,p=mcn(U[a],U[b]); m,c=ci(U[a],U[b]); out[f'{a}-{b}']=dict(gain=g,loss=l,p=p,diff=m,ci=c.tolist())
        print(f'{a} - {b}: 无缩进 {U[a].mean():.3f} vs {U[b].mean():.3f}  差 {m:+.3f} [{c[0]:+.3f},{c[1]:+.3f}]  不一致题 {g}/{l} p={p:.4f}')
json.dump(out,open('g/unindent_paired_treset.json','w'),indent=1)
