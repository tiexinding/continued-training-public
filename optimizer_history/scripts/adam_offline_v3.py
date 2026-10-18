#!/usr/bin/env python3
"""P1：在 adam_offline_v2 基础上增加因子对照——仅行因子 rowonly（log v = μ + a_row）、仅列因子 colonly（μ + b_col）、完整 rowcol、分布匹配随机因子 randfac（把 a_row、b_col 各自打乱后重组：保边际分布、破身份），
并对 gate/up（行 = 中间神经元）与 down（列 = 中间神经元）三种矩阵分别报"中间神经元轴"的份额。其余口径与 v2 相同（固定置换、逐批/总平方/合成、匹配前后、cos/相对误差/秩相关）。"""
import sys, re
src = open(__file__.replace("adam_offline_v3.py", "adam_offline_v2.py")).read()
# 扩展变体解析：rowonly/colonly 直接用 v_ladder 的 row/col；randfac 需新实现
src = src.replace('ap.add_argument("--variants", default="orig,rowcol,cperm,rperm,cperm_col,perm,const,fresh,perm_resample")',
                  'ap.add_argument("--variants", default="orig,rowcol,rowonly,colonly,randfac,cperm,rperm,cperm_col,const,fresh")')
src = src.replace('''def make_v(ver, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if ver in ("orig", "perm_resample"): return {n: vT[n].clone() for n, _ in W}
    if ver == "fresh": return {n: torch.zeros_like(vT[n]) for n, _ in W}
    return {n: fl.v_ladder(vT[n], ver, gen=gen) for n, _ in W}''',
'''def randfac(v, gen):
    L = torch.log(v.double().clamp_min(fl.V_FLOOR)); mu = L.mean(); a = L.mean(1) - mu; b = L.mean(0) - mu
    a = a[torch.randperm(a.numel(), generator=gen)]; b = b[torch.randperm(b.numel(), generator=gen)]
    return torch.exp(mu + a[:, None] + b[None, :]).to(v.dtype)
def make_v(ver, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if ver in ("orig", "perm_resample"): return {n: vT[n].clone() for n, _ in W}
    if ver == "fresh": return {n: torch.zeros_like(vT[n]) for n, _ in W}
    if ver == "rowonly": return {n: fl.v_ladder(vT[n], "row", gen=gen) for n, _ in W}
    if ver == "colonly": return {n: fl.v_ladder(vT[n], "col", gen=gen) for n, _ in W}
    if ver == "randfac": return {n: randfac(vT[n], gen) for n, _ in W}
    return {n: fl.v_ladder(vT[n], ver, gen=gen) for n, _ in W}''')
src = src.replace('seeds = range(a.perm_seeds) if ver in ("cperm", "rperm", "cperm_col", "perm", "perm_resample") else [0]',
                  'seeds = range(a.perm_seeds) if ver in ("cperm", "rperm", "cperm_col", "perm", "perm_resample", "randfac") else [0]')
# 三种矩阵的"中间神经元轴"份额：gate/up 用行能量（dim=0 → 行），down 用列（dim=1）——v2 的 KINDS 已是 (gate,0),(up,0),(down,1)，即都按中间神经元轴聚合。只需在打印中标明。
src = src.replace('print(f"{key:16} down(匹配后)', 'print(f"{key:16} down列=中间神经元(匹配后)')
exec(compile(src, "adam_offline_v3_inline", "exec"))
