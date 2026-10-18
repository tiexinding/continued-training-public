#!/usr/bin/env python3
"""生成 structure_by_kind_six_arms.json（10-16 恢复，文档 49 L5）。原为一次性命令，时间 2026-10-07T03:49:21.155Z（UTC），
从当时的会话记录逐字取回；此处只加本文件头与工作目录设置（原命令在该目录下运行），其余代码未改动，
个别输入路径的替换见下方 L5 注释。"""
import os as _os
from pathlib import Path as _P
_os.chdir(_P(__file__).resolve().parents[2])
# ---- 以下为原命令
import json,numpy as np
def kind(n):
    for k in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","embed"):
        if k in n: return k.replace("_proj","")
arms={'K_mvt':'DATA/t2/diag25/t2/lr5e-6/pairs/A_B.json','K_vt':'DATA/b2/pairs_Kv/A_B.json','K_t':'DATA/b2/pairs_bridge/A_B.json','K_∅':'DATA/t3/diag25/t3/pairs/A_B.json','C_mvt':'DATA/t2/diag25/t2/lr5e-6/pairs/A_contA.json','C_∅':'DATA/b2/pairs_Cfresh/A_B.json'}
KN=["q","k","v","o","gate","up","down","embed"]
print("== 六臂按矩阵种类的读数（24 层中位）：行固定斜率 R² | 列固定斜率 R² | cvZ/零(行) | ε")
out={}
for a,p in arms.items():
    pm=json.load(open(p))['per_matrix']; out[a]={}
    row=[]
    for k in KN:
        rs=[x for n,x in pm.items() if kind(n)==k]
        r2r=np.median([x['L3']['eqstep_Zrow_r2fixed'] for x in rs]); r2c=np.median([x['L3']['eqstep_Zcol_r2fixed'] for x in rs]); cv=np.median([x['L3']['shuffle']['cvZ_row']/x['L3']['shuffle']['cvZ_row_null_Wshuffle'] for x in rs]); eps=np.median([x['L3']['eps'] for x in rs])
        out[a][k]=dict(r2row=float(r2r),r2col=float(r2c),cv=float(cv),eps=float(eps)); row.append(f"{k}:{r2r:.2f}/{r2c:.2f}/{cv:.2f}")
    print(f"{a:6} "+"  ".join(row))
json.dump(out,open('DATA/b2/structure_by_kind_six_arms.json','w'),ensure_ascii=False,indent=1)
# 改动能量按种类的占比（用 fields npz 的 eW_row 之和；K_mvt 与 K_∅）
print("\n== 改动能量按矩阵种类占比（全模型 2D 矩阵；K_mvt 对 K_∅）")
for a,p in (('K_mvt','DATA/t2/fields/A_B_fields.npz'),('K_∅','DATA/t3/fields/A_B_fields.npz')):
    F=np.load(p); tot={k:0.0 for k in KN}
    for key in F.files:
        if key.endswith('|eW_row'):
            n=key.split('|')[0]; k=kind(n)
            if k: tot[k]+=float(F[key].sum())
    T=sum(tot.values()); print(f"{a:6} "+"  ".join(f"{k}:{tot[k]/T*100:.1f}%" for k in KN)+f"   前馈合计 {(tot['gate']+tot['up']+tot['down'])/T*100:.1f}%  注意力合计 {(tot['q']+tot['k']+tot['v']+tot['o'])/T*100:.1f}%")
