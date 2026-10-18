#!/usr/bin/env python3
"""10-11 J 轮（文档 46 第 15 节）：在卡上用卡上的 forklib.epoch_order 生成步序文件（避免本地/卡上 torch 版本差异）。
  J0  种子 17 原顺序（= 默认 epoch_order；用来核对 --order_file 路径逐位复现 E_vt / PG_vt_s17）
  J1  共享末段：最后 25 步 = 种子 17 的最后 25 步（同一组 200 条、同样顺序）；前 100 步 = 种子 s 的排列去掉这 200 条后按原相对次序每 8 条一步（s = 31、47）
用法：python make_orders_j.py --fork_dir <含 forklib.py 的目录> --n 1000 --out <目录>"""
import argparse, json, sys, hashlib
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--fork_dir", required=True); ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--out", required=True); ap.add_argument("--tail", type=int, default=25); a = ap.parse_args()
sys.path.insert(0, a.fork_dir); import forklib as fl
O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
base = fl.epoch_order(a.n, 1, 17); T = len(base); assert T == 125 and all(len(b) == 8 for b in base)
tail = base[T - a.tail:]; tailset = {i for b in tail for i in b}
def dump(name, order):
    flat = [i for b in order for i in b]; assert sorted(flat) == list(range(a.n)) and len(order) == T, name
    json.dump(order, open(O / f"{name}.json", "w")); print(name, hashlib.sha256(json.dumps(order).encode()).hexdigest()[:16])
dump("order_J0_s17", base)
for s in (31, 47):
    perm = [i for b in fl.epoch_order(a.n, 1, s) for i in b]; head = [i for i in perm if i not in tailset]
    dump(f"order_J1_s{s}", [head[i:i + 8] for i in range(0, len(head), 8)] + tail)
