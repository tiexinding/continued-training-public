#!/usr/bin/env python3
"""生成 intervene_v3_paired.json（10-16 恢复，文档 49 L5）。原为一次性命令，时间 2026-10-07T03:11:39.802Z（UTC），
从当时的会话记录逐字取回；此处只加本文件头与工作目录设置（原命令在该目录下运行），其余代码未改动，
个别输入路径的替换见下方 L5 注释。"""
import os as _os
from pathlib import Path as _P
_os.chdir(_P(__file__).resolve().parents[2])
# ---- 以下为原命令
import json,numpy as np
J=json.load(open('DATA/d/diag25/d5/intervene_Gkeep_v3.json')); B=J['B']; rng=np.random.default_rng(0)
# 逐题等权 vs token 加权（token 加权用聚合 nll 差）
print("配置 | 撤回能量% | token 加权回退 | 逐题等权回退 [95%] | 对 sel_mbpp 的逐题直接差 [95%]")
Bi=np.array(B['nll_per_item']['probe_mbpp']); Si=np.array(J['sel_mbpp@0.05']['nll_per_item']['probe_mbpp'])
out={}
for k in ['sel_mbpp@0.05']+[f'mbpp_dual_matched{r}@0.05' for r in range(5)]+[f'mbpp_matched_random{r}@0.05' for r in range(5)]+['random@0.05','energy@0.05','interp@0.05']:
    v=J[k]; tw=B['nll']['probe_mbpp']-v['nll']['probe_mbpp']; d=Bi-np.array(v['nll_per_item']['probe_mbpp']); bs=[rng.choice(d,len(d)).mean() for _ in range(4000)]
    dd=(Bi-Si)-d; bs2=[rng.choice(dd,len(dd)).mean() for _ in range(4000)]
    out[k]=dict(e=v['reverted_energy_frac'],tok=tw,item=float(d.mean()),item_ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],diff_vs_sel=float(dd.mean()),diff_ci=[float(np.percentile(bs2,2.5)),float(np.percentile(bs2,97.5))])
    print(f"{k:26} | {v['reverted_energy_frac']*100:.3f} | {tw:+.5f} | {d.mean():+.5f} [{np.percentile(bs,2.5):+.5f},{np.percentile(bs,97.5):+.5f}] | {dd.mean():+.5f} [{np.percentile(bs2,2.5):+.5f},{np.percentile(bs2,97.5):+.5f}]")
json.dump(out,open('DATA/d/intervene_v3_paired.json','w'),indent=1)
# 优势倍数（两种口径）
sel=out['sel_mbpp@0.05']; dual=[out[f'mbpp_dual_matched{r}@0.05'] for r in range(5)]
print("倍数（逐题）:", [round(sel['item']/d['item'],2) for d in dual], " 倍数（token）:", [round(sel['tok']/d['tok'],2) for d in dual])
print("双匹配能量/目标能量：", [round(d['e']/sel['e'],3) for d in dual])
# HumanEval 规范解 NLL：含 prompt 复制段，无法在现有 json 拆分（只存了逐题总 NLL）→ 标注
print("HumanEval 规范解 NLL 回退（sel_mbpp, interp）：", round(B['nll']['heldout_humaneval']-J['sel_mbpp@0.05']['nll']['heldout_humaneval'],5), round(B['nll']['heldout_humaneval']-J['interp@0.05']['nll']['heldout_humaneval'],5))
