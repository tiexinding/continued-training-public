#!/usr/bin/env python3
"""生成 neuron_overlap_sixarms.json（10-16 恢复，文档 49 L5）。原为一次性命令，时间 2026-10-06T05:46:31.243Z（UTC），
从当时的会话记录逐字取回；此处只加本文件头与工作目录设置（原命令在该目录下运行），其余代码未改动，
个别输入路径的替换见下方 L5 注释。"""
import os as _os
from pathlib import Path as _P
_os.chdir(_P(__file__).resolve().parents[2] / "DATA")
# ---- 以下为原命令
import json,numpy as np
A=np.load('b4/neuron_act_M.npz'); J=json.load(open('b4/neuron_act_M.json'))
print("层级：GSM8K 与 MBPP 激活 RMS 相关（逐层）:", [round(J['layers'][f'L{i}']['rms_corr_gsm_mbpp'],2) for i in range(24)])
def topset(x,frac=0.05): k=max(1,int(frac*len(x))); return set(np.argsort(-x)[:k].tolist()),k
fields={'Gkeep':'t2/fields/A_B_fields.npz','Kv':'b2/fields/Kv_A_B_fields.npz','bridge':'b2/fields/bridge_A_B_fields.npz','Gfresh':'t3/fields/A_B_fields.npz','Ckeep':'b2/fields/Cfresh_A_contA_fields.npz','Cfresh':'b2/fields/Cfresh_A_B_fields.npz'}
out={}
print("\n臂 | down列能量 top5%∩GSM8K选择性 | ∩MBPP选择性 | ρ(down能量,GSM8K激活) | ρ(down能量,MBPP激活) | up行能量∩GSM8K选择性 | ρ(up能量,MBPP激活) | 逐层 down∩GSM8K选择性（0,6,12,18,23）")
for a,p in fields.items():
    F=np.load(p); rows=[]
    for i in range(24):
        g=A[f'L{i}_gsm8k_rms']; m=A[f'L{i}_mbpp_rms']; sel=A[f'L{i}_sel_gsm_vs_mbpp']
        d=F[f'model.layers.{i}.mlp.down_proj.weight|eW_col']; u=F[f'model.layers.{i}.mlp.up_proj.weight|eW_row']
        S,k=topset(d); Tg,_=topset(sel); Tm,_=topset(-sel); Su,_=topset(u)
        rk=lambda x: np.argsort(np.argsort(x))
        rows.append(dict(dg=len(S&Tg)/k, dm=len(S&Tm)/k, rg=np.corrcoef(rk(d),rk(g))[0,1], rm=np.corrcoef(rk(d),rk(m))[0,1], ug=len(Su&Tg)/k, urm=np.corrcoef(rk(u),rk(m))[0,1]))
    med={k:float(np.median([r[k] for r in rows])) for k in rows[0]}; out[a]=dict(median=med,per_layer=rows)
    print(f"{a:7} | {med['dg']:.3f} | {med['dm']:.3f} | {med['rg']:+.3f} | {med['rm']:+.3f} | {med['ug']:.3f} | {med['urm']:+.3f} | "+" ".join(f"{rows[i]['dg']:.2f}" for i in (0,6,12,18,23)))
json.dump(out,open('b4/neuron_overlap_sixarms.json','w'),ensure_ascii=False,indent=1)
