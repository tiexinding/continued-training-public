#!/usr/bin/env python3
"""生成 predictor_baselines_dir2.json（10-16 恢复，文档 49 L5）。原为一次性命令，时间 2026-10-07T03:12:10.903Z（UTC），
从当时的会话记录逐字取回；此处只加本文件头与工作目录设置（原命令在该目录下运行），其余代码未改动，
个别输入路径的替换见下方 L5 注释。"""
import os as _os
from pathlib import Path as _P
_os.chdir(_P(__file__).resolve().parents[2] / "DATA/d")
# ---- 以下为原命令
import numpy as np, json
A=np.load('diag25/b4/neuron_act_A_g_d.npz'); F=np.load('fields/../fields/R_vt_A_B_fields.npz') if False else None
# 第二方向预测基线：目标 = R_vt 终态 down 列能量 top-5%（新任务 = MBPP）
import glob
# 终态 down 列能量在 pairs json 不含逐通道；用 records 的 dW 终点（step0108）作替代
import torch
def top(a,b,frac=.05): k=max(1,int(frac*len(a))); return len(set(np.argsort(-a)[:k])&set(np.argsort(-b)[:k]))/k
rec={s:torch.load(f'records/R_vt/fields_{s:04d}.pt',map_location='cpu',weights_only=False)['fields'] for s in (2,10,100)}
res={}
for i in range(24):
    n=f'model.layers.{i}.mlp.down_proj.weight'; fin=np.exp(rec[100][n]['dW'][1].numpy().astype(float))
    g=A[f'L{i}_gsm8k_rms']; m=A[f'L{i}_mbpp_rms']
    v2=np.exp(rec[2][n]['vhat'][1].numpy().astype(float)); g2=np.exp(rec[2][n]['g'][1].numpy().astype(float)); dW10=np.exp(rec[10][n]['dW'][1].numpy().astype(float))
    preds={'旧任务低激活（−GSM8K RMS）':-g,'新任务高激活（MBPP RMS）':m,'两任务激活比 (M−G)/(M+G)':(m-g)/(m+g+1e-12),'旧 v 的倒数（1/v）':1/v2,'初始梯度能量':g2,'初始 g²/v':g2/v2,'第10步累计改动':dW10}
    for k,p in preds.items(): res.setdefault(k,[]).append(top(p,fin))
print("第二方向（R_vt 第 100 步累计 down 列能量 top-5% 命中率，24 层中位，随机 .05）")
for k,v in res.items(): print(f"{k:28} {np.median(v):.2f} [{min(v):.2f},{max(v):.2f}]")
json.dump({k:float(np.median(v)) for k,v in res.items()},open('predictor_baselines_dir2.json','w'),ensure_ascii=False,indent=1)
