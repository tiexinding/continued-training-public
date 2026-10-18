#!/usr/bin/env python3
"""K 轮 legacy_check（文档 47 v2 第 4 节第 3 步）：重跑的 adam_offline_v3 与文档 36 原 JSON 逐项比较（各变体 raw/matched 的 down/gate/up 中位数）。
份额类键（*share*）最大绝对差 > 0.003（0.3 个百分点）即记"不一致"，只报告、不调整定义。用法：python compare_legacy.py --new <json> --ref <json> --out <json>"""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("--new", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
N, R = json.load(open(a.new)), json.load(open(a.ref)); out = {"variants": {}, "max_share_absdiff": 0.0, "bitwise_equal_all": True}
for v in R["variants"]:
    if v not in N["variants"]: out["variants"][v] = "缺"; continue
    for sc in ("raw", "matched"):
        for k in R["variants"][v][sc]["median"]:
            for q, x in R["variants"][v][sc]["median"][k].items():
                y = N["variants"][v][sc]["median"][k][q]; d = abs(x - y)
                out["bitwise_equal_all"] &= (x == y)
                if "share" in q: out["max_share_absdiff"] = max(out["max_share_absdiff"], d)
                out["variants"].setdefault(v, {}).setdefault(sc, {}).setdefault(k, {})[q] = [x, y]
out["consistent"] = out["max_share_absdiff"] <= 0.003
m = out["variants"]["orig"]["matched"]["down_proj"]
print("legacy: 份额最大绝对差", out["max_share_absdiff"], "逐位相同", out["bitwise_equal_all"], "| down orig 新/旧 GSM 份额", m["perbatch_share_gsm_mean"])
json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
