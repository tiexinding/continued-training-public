#!/usr/bin/env python3
"""生成 replay_budget_audit.json（10-16 恢复，文档 49 L5）。原为一次性命令，时间 2026-10-07T06:26:33.681Z（UTC），
从当时的会话记录逐字取回；此处只加本文件头与工作目录设置（原命令在该目录下运行），其余代码未改动，
个别输入路径的替换见下方 L5 注释。"""
import os as _os
from pathlib import Path as _P
_os.chdir(_P(__file__).resolve().parents[2])
# ---- 以下为原命令
# 回放预算审计：按训练口径（chat 模板 + 完成掩码 + max_len 512 截断）统计 125 步实际消费的新/旧域 token
import json,sys
sys.path.insert(0,'train'); import forklib as fl
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(_os.environ.get('OH_TOKENIZER','Qwen/Qwen2.5-0.5B-Instruct'))   # L5：原为本地分词器副本路径
P=_os.environ['OH_REPLAY_FILE']   # L5：原为当时会话临时目录中的回放文件；由 replay_data.py 重建（见 DATA_PROVENANCE）
rows=fl.prompt_completion_rows(P,tok,512); raw=[json.loads(l) for l in open(P)]
print("prompt_completion_rows 保留", len(rows), "/", len(raw), "（过滤超长后）")
exs=fl.tokenize_train_manual(rows,tok,512)
# 对应回放标记：rows 与 raw 顺序一致？用 prompt 文本匹配
is_old=[]
rawq={r['messages'][0]['content']:r.get('metadata',{}).get('replay',False) for r in raw}
for r in rows:
    q=r['prompt']; flag=None
    for k,v in rawq.items():
        if k in q: flag=v; break
    is_old.append(bool(flag))
order=fl.epoch_order(len(exs),1,17)[:125]
inp={'old':0,'new':0}; sup={'old':0,'new':0}; n={'old':0,'new':0}; trunc=0
for st in order:
    for i in st:
        e=exs[i]; k='old' if is_old[i] else 'new'; inp[k]+=len(e.input_ids); sup[k]+=sum(e.completion_mask); n[k]+=1
        if len(e.input_ids)>=512: trunc+=1
ti=inp['old']+inp['new']; ts=sup['old']+sup['new']
print(f"125 步实际消费：样本 新 {n['new']} / 旧 {n['old']}；输入 token 新 {inp['new']} / 旧 {inp['old']}（旧占 {inp['old']/ti*100:.1f}%）；监督 token 新 {sup['new']} / 旧 {sup['old']}（旧占 {sup['old']/ts*100:.1f}%）；截断样本 {trunc}")
json.dump(dict(n=n,input_tokens=inp,supervised_tokens=sup,old_frac_input=inp['old']/ti,old_frac_sup=sup['old']/ts,truncated=trunc,steps=125,kept_rows=len(rows),raw_rows=len(raw)),open(_os.environ.get('OH_OUT','DATA/e/replay_budget_audit.json'),'w'),indent=1)
