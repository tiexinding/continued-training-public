#!/usr/bin/env python3
"""生成 early_checkpoint_W4_4.json（10-16 恢复，文档 49 L5）。原为一次性命令，时间 2026-10-06T05:49:50.053Z（UTC），
从当时的会话记录逐字取回；此处只加本文件头与工作目录设置（原命令在该目录下运行），其余代码未改动，
个别输入路径的替换见下方 L5 注释。"""
import os as _os
from pathlib import Path as _P
_os.chdir(_P(__file__).resolve().parents[2] / "DATA")
# ---- 以下为原命令
import torch,numpy as np,json
A=np.load('b4/neuron_act_M.npz')
steps=[1,2,5,10,25,50,75,100,125]
def topset(x,frac=0.05): k=max(1,int(frac*len(x))); return set(np.argsort(-x)[:k].tolist()),k
rk=lambda x: np.argsort(np.argsort(x)); sp=lambda a,b: float(np.corrcoef(rk(a),rk(b))[0,1])
out={}
for arm in ('Kv','bridge'):
    recs={s:torch.load(f'b2/records/runs_b2/{arm}/records/fields_{s:04d}.pt',map_location='cpu',weights_only=False)['fields'] for s in steps}
    fin={i:recs[125][f'model.layers.{i}.mlp.down_proj.weight']['dW'][1].numpy() for i in range(24)}
    print(f"== {arm}  （down 列；24 层中位）")
    print(" 步 | vhat列∩GSM8K选择性 | ρ(vhat列,MBPP激活) | ρ(vhat列,GSM8K激活) | u列∩GSM8K选择性 | ρ(u列,MBPP激活) | dW列∩GSM8K选择性 | ρ(dW_s, dW_终) | dW_s top5%∩dW_终 top5%")
    rows={}
    for s in steps:
        r=[]
        for i in range(24):
            f=recs[s][f'model.layers.{i}.mlp.down_proj.weight']; g=A[f'L{i}_gsm8k_rms']; m=A[f'L{i}_mbpp_rms']; sel=A[f'L{i}_sel_gsm_vs_mbpp']
            Tg,k=topset(sel); vh=f['vhat'][1].numpy(); u=f['u'][1].numpy(); dW=f['dW'][1].numpy()
            Sv,_=topset(vh); Su,_=topset(u); Sd,_=topset(dW); Sf,_=topset(fin[i])
            r.append([len(Sv&Tg)/k, sp(vh,m), sp(vh,g), len(Su&Tg)/k, sp(u,m), len(Sd&Tg)/k, sp(dW,fin[i]), len(Sd&Sf)/k])
        med=np.median(np.array(r),0); rows[s]=med.tolist()
        print(f" {s:3} | "+" | ".join(f"{x:+.3f}" for x in med))
    out[arm]=rows
json.dump(out,open('b4/early_checkpoint_W4_4.json','w'),indent=1)
