#!/usr/bin/env python3
"""pairlib 的 torch 后端（GPU/CPU 通用），接口与输出键与 pairlib.analyse_pair 完全一致。
默认 float64（与 numpy 版逐位可比；validate_backend.py 核验）；dtype=torch.float32 可再快一倍，闭环 ~1e-6。
用法：import pairlib_t as pl; pl.DEVICE = "cuda"; res, fields = pl.analyse_pair(WA, WB, rng=np.random.default_rng(0))
WA、WB 可为 numpy 数组或 torch 张量；fields 返回 numpy 数组。"""
from __future__ import annotations
import math
import numpy as np, torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64          # 分解、几何、场：float64（下界角 0.06° 级的量需要）
SPEC_DTYPE = torch.float32    # 奇异值、子空间、主子空间重合：float32（GPU 上快 10 倍，相对误差 ~1e-6）
EPS = 1e-300


def _t(x):
    return x.to(DEVICE, DTYPE) if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), dtype=DTYPE, device=DEVICE)


def _np(x):
    return x.detach().cpu().numpy()


def _f(x):
    return float(x)


def rms(W):
    return _f(torch.sqrt((_t(W) ** 2).mean()))


# ---------------------------------------------------------------- 分解
CONV = []   # 10-09 H2′：每次 decompose 的迭代次数与最终行/列 RMS 变异系数（调用方按矩阵清空、读取）


def decompose(W, it: int = 500, tol: float = 1e-7):
    W = _t(W); m, n = W.shape
    s = torch.sqrt((W ** 2).mean())
    a = torch.ones(m, dtype=DTYPE, device=DEVICE); b = torch.ones(n, dtype=DTYPE, device=DEVICE); z = W / s
    for _ in range(it):
        r = torch.sqrt((z ** 2).mean(1)); z = z / r.clamp_min(EPS)[:, None]; a = a * r
        c = torch.sqrt((z ** 2).mean(0)); z = z / c.clamp_min(EPS)[None, :]; b = b * c
        rr = torch.sqrt((z ** 2).mean(1)); cc = torch.sqrt((z ** 2).mean(0))
        if _f(rr.std(unbiased=False) / rr.mean()) < tol and _f(cc.std(unbiased=False) / cc.mean()) < tol:
            break
    CONV.append(dict(shape=[m, n], iters=_ + 1, cv_row=_f(rr.std(unbiased=False) / rr.mean()), cv_col=_f(cc.std(unbiased=False) / cc.mean()), tol=tol))   # 10-09 H2′：收敛记录
    la, lb = torch.log(a.clamp_min(EPS)), torch.log(b.clamp_min(EPS))
    g = 0.5 * (la.mean() - lb.mean()); hr, hc = la - g, lb + g
    lvl = 0.5 * (hr.mean() + hc.mean()); hr, hc = hr - lvl, hc - lvl; s = s * torch.exp(2 * lvl)
    rec = s * torch.exp(hr)[:, None] * z * torch.exp(hc)[None, :]
    err = _f((rec - W).abs().max() / W.abs().max().clamp_min(EPS))
    assert err < 1e-6 if DTYPE == torch.float32 else err < 1e-10, ("W = s D_r Z D_c violated", err)
    return s, hr, hc, z


def delta_z_ref(WA, WB, sA, hrA, hcA):
    WA, WB = _t(WA), _t(WB); dW = WB - WA
    return dW / (sA * torch.exp(hrA)[:, None] * torch.exp(hcA)[None, :]), dW


# ---------------------------------------------------------------- 几何
def _rowgeo(A, D):
    na2 = (A ** 2).sum(1); nd2 = (D ** 2).sum(1); dot = (A * D).sum(1)
    rho = torch.sqrt(nd2 / na2.clamp_min(EPS)); p = dot / na2.clamp_min(EPS)
    q = torch.sqrt((nd2 / na2.clamp_min(EPS) - p ** 2).clamp_min(0.0))
    nb2 = na2 + 2 * dot + nd2; cos = (na2 + dot) / torch.sqrt((na2 * nb2).clamp_min(EPS))
    return dict(rho=rho, par=p, perp=q, theta=torch.arccos(cos.clamp(-1, 1)), eA2=na2, eD2=nd2)


def row_geometry_W(WA, dW): return _rowgeo(WA, dW)


def row_geometry_Zweighted(ZA, dZ, hcA):
    w = torch.exp(hcA)[None, :]; return _rowgeo(ZA * w, dZ * w)


def row_geometry_Zplain(ZA, dZ): return _rowgeo(ZA, dZ)


def matrix_geometry(WA, dW):
    A2 = (WA ** 2).sum(); D2 = (dW ** 2).sum(); dot = (WA * dW).sum()
    eps = math.sqrt(_f(D2 / A2)); par = _f(dot / A2); par_energy = _f(dot ** 2 / A2 / D2)
    B2 = A2 + 2 * dot + D2; cosT = _f((A2 + dot) / torch.sqrt(A2 * B2))
    return dict(eps=eps, Theta=math.degrees(math.acos(max(-1, min(1, cosT)))), par_rel=par, par_energy_frac=par_energy, perp_energy_frac=1 - par_energy,
                norm_rel_change=math.sqrt(_f(B2 / A2)) - 1, cos_dW_WA=_f(dot / torch.sqrt(A2 * D2)))


def length_vector_angles(WA, WB):
    out = {}
    for name, ax in (("row", 1), ("col", 0)):
        la = torch.sqrt((WA ** 2).sum(ax)); lb = torch.sqrt((WB ** 2).sum(ax))
        c = _f((la * lb).sum() / torch.sqrt((la ** 2).sum() * (lb ** 2).sum()))
        out[f"Theta_{name}_len_deg"] = math.degrees(math.acos(min(1, c)))
    return out


def fit_rowcol(Wf, Wt, iters=20):
    r = torch.ones(Wf.shape[0], dtype=DTYPE, device=DEVICE); c = torch.ones(Wf.shape[1], dtype=DTYPE, device=DEVICE)
    for _ in range(iters):
        X = Wf * c[None, :]; r = (Wt * X).sum(1) / (X ** 2).sum(1).clamp_min(EPS)
        X = Wf * r[:, None]; c = (Wt * X).sum(0) / (X ** 2).sum(0).clamp_min(EPS)
    return r, c


def scaling_explained(WA, WB, sA, hrA, hcA, sB, hrB, hcB):
    dW = WB - WA; Etot = (dW ** 2).sum()

    def three(S, tag):
        U, V = S - WA, WB - S; u2, v2 = _f((U ** 2).sum() / Etot), _f((V ** 2).sum() / Etot)
        return {f"subst_{tag}": u2, f"res_{tag}": v2, f"cross_{tag}": 1 - u2 - v2, f"resid_drop_{tag}": 1 - v2}

    r, c = fit_rowcol(WA, WB); a = (WB * WA).sum() / (WA ** 2).sum()
    out = dict(**three(r[:, None] * WA * c[None, :], "ls"), **three(a * WA, "global"))
    Sf = (sB / sA) * torch.exp(hrB - hrA)[:, None] * WA * torch.exp(hcB - hcA)[None, :]
    out.update(three(Sf, "field")); return out


# ---------------------------------------------------------------- 能量场、零模型、零假设
def equal_step_baseline(e_row, h_row, fixed_slope=-2.0):
    y = torch.log(e_row.clamp_min(EPS)); x = h_row
    X = torch.stack([x, torch.ones_like(x)], 1); beta = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
    res = y - X @ beta; sst = ((y - y.mean()) ** 2).sum(); r2 = 1 - (res ** 2).sum() / sst
    pf = fixed_slope * x; pf = pf - pf.mean() + y.mean(); r2f = 1 - ((y - pf) ** 2).sum() / sst
    return dict(slope=_f(beta[0]), r2_fit=_f(r2), r2_fixed=_f(r2f), resid_fit=res, resid_fixed=y - pf)


def shuffle_null(dZ, dW, sA, hrA, hcA, rng, reps=5):
    m, n = dZ.shape; scale = sA * torch.exp(hrA)[:, None] * torch.exp(hcA)[None, :]
    gen = torch.Generator(device=DEVICE); gen.manual_seed(int(rng.integers(0, 2 ** 31)))
    acc = {k: [] for k in ("cvZr_W", "cvZc_W", "cvWr", "cvWc", "cvZr_Z", "cvZc_Z", "r2fit", "r2fix")}; last = None
    flatW, flatZ = dW.reshape(-1), dZ.reshape(-1)

    def cv(v): return _f(v.std(unbiased=False) / v.mean())

    for _ in range(reps):
        pW = flatW[torch.randperm(m * n, generator=gen, device=DEVICE)].view(m, n); pZ = pW / scale; last = pZ
        er, ec = (pZ ** 2).sum(1), (pZ ** 2).sum(0); acc["cvZr_W"].append(cv(er)); acc["cvZc_W"].append(cv(ec))
        b = equal_step_baseline(er, hrA); acc["r2fit"].append(b["r2_fit"]); acc["r2fix"].append(b["r2_fixed"])
        er, ec = (pW ** 2).sum(1), (pW ** 2).sum(0); acc["cvWr"].append(cv(er)); acc["cvWc"].append(cv(ec))
        p = flatZ[torch.randperm(m * n, generator=gen, device=DEVICE)].view(m, n); er, ec = (p ** 2).sum(1), (p ** 2).sum(0)
        acc["cvZr_Z"].append(cv(er)); acc["cvZc_Z"].append(cv(ec))
    er, ec = (dZ ** 2).sum(1), (dZ ** 2).sum(0); ewr, ewc = (dW ** 2).sum(1), (dW ** 2).sum(0)
    mean = lambda k: float(np.mean(acc[k]))
    return dict(cvZ_row=cv(er), cvZ_col=cv(ec), cvZ_row_null_Wshuffle=mean("cvZr_W"), cvZ_col_null_Wshuffle=mean("cvZc_W"),
                cvW_row=cv(ewr), cvW_col=cv(ewc), cvW_row_null=mean("cvWr"), cvW_col_null=mean("cvWc"), cvW_row_gauss=math.sqrt(2 / n), cvW_col_gauss=math.sqrt(2 / m),
                cvZ_row_null_Zshuffle=mean("cvZr_Z"), cvZ_col_null_Zshuffle=mean("cvZc_Z"), eqstep_r2fit_null=mean("r2fit"), eqstep_r2fixed_null=mean("r2fix"), _null_dZ=last)


def _svdvals(D):
    return torch.linalg.svdvals(D.to(SPEC_DTYPE)).to(DTYPE)


def spectrum(D, rng=None):
    m, n = D.shape; Etot = (D ** 2).sum(); sv = _svdvals(D); sv2 = sv ** 2
    edge = math.sqrt(_f(Etot) / (m * n)) * (math.sqrt(m) + math.sqrt(n)); above = sv > edge
    out = dict(n_above_mp=int(above.sum()), energy_above_mp=_f(sv2[above].sum() / sv2.sum()), erank=_f(sv2.sum() ** 2 / (sv2 ** 2).sum()),
               top1_frac=_f(sv2[0] / sv2.sum()), top10_frac=_f(sv2[:10].sum() / sv2.sum()))
    if rng is not None:
        gen = torch.Generator(device=DEVICE); gen.manual_seed(int(rng.integers(0, 2 ** 31)))
        R = torch.randn(m, n, generator=gen, device=DEVICE, dtype=DTYPE) * math.sqrt(_f(Etot) / (m * n)); rs = _svdvals(R); rs2 = rs ** 2
        out.update(rand_n_above_mp=int((rs > edge).sum()), rand_top10_frac=_f(rs2[:10].sum() / rs2.sum()), rand_erank=_f(rs2.sum() ** 2 / (rs2 ** 2).sum()))
    return out


def _topk_svd(Z, r):
    """前 r 个左/右奇异向量：m ≫ n 时走 Gram 矩阵（n×n）特征分解，否则完整 SVD。"""
    m, n = Z.shape; Z = Z.to(SPEC_DTYPE)
    if m >= 4 * n:
        G = (Z.T @ Z).to(torch.float64); ev, V = torch.linalg.eigh(G); V = V.to(SPEC_DTYPE); ev = ev.to(SPEC_DTYPE); idx = torch.argsort(ev, descending=True)[:r]; V = V[:, idx]
        U = Z @ V / torch.sqrt(ev[idx]).clamp_min(1e-30)[None, :]; return U.to(DTYPE), V.T.to(DTYPE)
    U, S, Vt = torch.linalg.svd(Z, full_matrices=False); return U[:, :r].to(DTYPE), Vt[:r].to(DTYPE)


def subspace_share(ZA, dZ, ranks=(8, 32, 128), null_dZ=None):
    m, n = ZA.shape; rmax = max(x for x in ranks if x < min(m, n)); U, Vt = _topk_svd(ZA, rmax)
    E = (dZ ** 2).sum(); En = (null_dZ ** 2).sum() if null_dZ is not None else None; out = {}
    for r in ranks:
        if r >= min(m, n): continue
        out[f"left_r{r}"] = _f(((U[:, :r].T @ dZ) ** 2).sum() / E); out[f"left_r{r}_null_iso"] = r / m
        out[f"right_r{r}"] = _f(((dZ @ Vt[:r].T) ** 2).sum() / E); out[f"right_r{r}_null_iso"] = r / n
        if null_dZ is not None:
            out[f"left_r{r}_null_shuf"] = _f(((U[:, :r].T @ null_dZ) ** 2).sum() / En); out[f"right_r{r}_null_shuf"] = _f(((null_dZ @ Vt[:r].T) ** 2).sum() / En)
    return out


def principal_overlap(ZA, ZB, ranks=(8, 32)):
    m, n = ZA.shape; rmax = max(ranks); UA, VtA = _topk_svd(ZA, rmax); UB, VtB = _topk_svd(ZB, rmax); out = {}
    for r in ranks:
        out[f"left_r{r}"] = _f(((UA[:, :r].T @ UB[:, :r]) ** 2).sum() / r); out[f"right_r{r}"] = _f(((VtA[:r] @ VtB[:r].T) ** 2).sum() / r)
        out[f"left_r{r}_null"] = r / m; out[f"right_r{r}_null"] = r / n
    return out


def core_shape(Z):
    x = Z.reshape(-1); x = x - x.mean(); m2 = (x ** 2).mean()
    return dict(kurt_excess=_f((x ** 4).mean() / m2 ** 2 - 3), skew=_f((x ** 3).mean() / m2 ** 1.5))


def _spearman(a, b):
    ra = torch.argsort(torch.argsort(a)).to(DTYPE); rb = torch.argsort(torch.argsort(b)).to(DTYPE)
    return _f(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def _med(x):
    """numpy 约定的中位数（偶数长度取中间两数平均）。"""
    x = x.reshape(-1); return torch.quantile(x, 0.5) if x.numel() < 16_000_000 else x.median()


def _corr(a, b): return _f(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _polyfit1(x, y):
    X = torch.stack([x, torch.ones_like(x)], 1); return torch.linalg.lstsq(X, y[:, None]).solution[:, 0]


def _r2_lin(x, y):
    b = _polyfit1(x, y); res = y - (b[0] * x + b[1]); return _f(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum())


# ---------------------------------------------------------------- 四层汇总（输出与 pairlib.analyse_pair 同键）
def analyse_pair(WA, WB, rng=None, ranks=(8, 32, 128), with_spectrum=True, keep_fields=True, shuffle_reps=5):
    WA, WB = _t(WA), _t(WB)
    sA, hrA, hcA, ZA = decompose(WA); sB, hrB, hcB, ZB = decompose(WB); rA, rB = _f(torch.sqrt((WA ** 2).mean())), _f(torch.sqrt((WB ** 2).mean()))
    dZ, dW = delta_z_ref(WA, WB, sA, hrA, hcA)
    rec = sA * torch.exp(hrA)[:, None] * (ZA + dZ) * torch.exp(hcA)[None, :]; closure = _f((rec - WB).abs().max() / WB.abs().max())
    L1 = dict(s_bal_A=_f(sA), s_bal_B=_f(sB), rms_A=rA, rms_B=rB, s_bal_over_rms_A=_f(sA) / rA, H_row_A=_f(hrA.std(unbiased=False)), H_col_A=_f(hcA.std(unbiased=False)),
              H_row_B=_f(hrB.std(unbiased=False)), H_col_B=_f(hcB.std(unbiased=False)), coreA=core_shape(ZA), coreB=core_shape(ZB))
    dhr, dhc = hrB - hrA, hcB - hcA
    L2 = dict(drms_rel=rB / rA - 1, ds_bal_rel=_f(sB / sA) - 1, dH_row=_f(hrB.std(unbiased=False) - hrA.std(unbiased=False)), dH_col=_f(hcB.std(unbiased=False) - hcA.std(unbiased=False)),
              sd_dh_row=_f(dhr.std(unbiased=False)), sd_dh_col=_f(dhc.std(unbiased=False)), corr_h_row=_corr(hrA, hrB), corr_h_col=_corr(hcA, hcB),
              spearman_h_row=_spearman(hrA, hrB), spearman_h_col=_spearman(hcA, hcB),
              dh_vs_h_slope_row=_f(_polyfit1(hrA, dhr)[0]), dh_vs_h_slope_col=_f(_polyfit1(hcA, dhc)[0]))
    gW = row_geometry_W(WA, dW); gWz = row_geometry_Zweighted(ZA, dZ, hcA); gZ = row_geometry_Zplain(ZA, dZ)
    gWc = row_geometry_W(WA.T, dW.T); gZc = row_geometry_Zplain(ZA.T, dZ.T)
    agree = _f((gW["theta"] - gWz["theta"]).abs().max())
    eZr, eZc, eWr, eWc = (dZ ** 2).sum(1), (dZ ** 2).sum(0), (dW ** 2).sum(1), (dW ** 2).sum(0)
    base_r = equal_step_baseline(eZr, hrA); base_c = equal_step_baseline(eZc, hcA)
    baseW_r = equal_step_baseline(eWr, hrA, 0.0); baseW_c = equal_step_baseline(eWc, hcA, 0.0)
    lnrowA = 0.5 * torch.log(gW["eA2"].clamp_min(EPS)); lncolA = 0.5 * torch.log(gWc["eA2"].clamp_min(EPS))
    th_r = equal_step_baseline(gW["theta"], hrA, -1.0); th_c = equal_step_baseline(gWc["theta"], hcA, -1.0)
    th_r_raw = equal_step_baseline(gW["theta"], lnrowA, -1.0); th_c_raw = equal_step_baseline(gWc["theta"], lncolA, -1.0)
    par_rows = _f((gW["par"] ** 2 * gW["eA2"]).sum() / gW["eD2"].sum()); par_cols = _f((gWc["par"] ** 2 * gWc["eA2"]).sum() / gWc["eD2"].sum())
    deg = lambda t: float(math.degrees(_f(t)))
    L3 = dict(**matrix_geometry(WA, dW), **length_vector_angles(WA, WB), **scaling_explained(WA, WB, sA, hrA, hcA, sB, hrB, hcB),
              theta_W_deg_med=deg(_med(gW["theta"])), theta_W_deg_p90=deg(torch.quantile(gW["theta"], 0.9)) if gW["theta"].numel() < 16_000_000 else deg(gW["theta"].kthvalue(int(0.9 * gW["theta"].numel())).values),
              theta_Z_deg_med=deg(_med(gZ["theta"])), theta_Wcol_deg_med=deg(_med(gWc["theta"])), theta_Zcol_deg_med=deg(_med(gZc["theta"])),
              par_rel_row_med=_f(_med(gW["par"])), perp_rel_row_med=_f(_med(gW["perp"])),
              par_energy_rows=par_rows, par_energy_cols=par_cols, par_energy_rows_signed_frac=_f((gW["par"] > 0).to(DTYPE).mean()),
              theta_W_vs_Zweighted_maxabs=agree, corr_theta_W_Z_row=_corr(gW["theta"], gZ["theta"]),
              eqstep_Zrow_slope=base_r["slope"], eqstep_Zrow_r2fit=base_r["r2_fit"], eqstep_Zrow_r2fixed=base_r["r2_fixed"],
              eqstep_Zcol_slope=base_c["slope"], eqstep_Zcol_r2fit=base_c["r2_fit"], eqstep_Zcol_r2fixed=base_c["r2_fixed"],
              eqstep_Wrow_slope=baseW_r["slope"], eqstep_Wrow_r2fit=baseW_r["r2_fit"], eqstep_Wcol_slope=baseW_c["slope"], eqstep_Wcol_r2fit=baseW_c["r2_fit"],
              theta_vs_h_slope_row=th_r["slope"], theta_vs_h_r2fit_row=th_r["r2_fit"], theta_vs_h_r2fixed_row=th_r["r2_fixed"],
              theta_vs_h_slope_col=th_c["slope"], theta_vs_h_r2fit_col=th_c["r2_fit"], theta_vs_h_r2fixed_col=th_c["r2_fixed"],
              theta_vs_lnrow_slope_row=th_r_raw["slope"], theta_vs_lnrow_r2fit_row=th_r_raw["r2_fit"], theta_vs_lnrow_r2fixed_row=th_r_raw["r2_fixed"],
              theta_vs_lncol_slope_col=th_c_raw["slope"], theta_vs_lncol_r2fit_col=th_c_raw["r2_fit"])
    null_dZ = None
    if rng is not None:
        sh = shuffle_null(dZ, dW, sA, hrA, hcA, rng, reps=shuffle_reps); null_dZ = sh.pop("_null_dZ"); L3["shuffle"] = sh
    if with_spectrum:
        L3["spec_dW"] = spectrum(dW, rng); L3["spec_dZ"] = spectrum(dZ, rng)
        L3["subspace"] = subspace_share(ZA, dZ, ranks, null_dZ); L3["principal_overlap_AB"] = principal_overlap(ZA, ZB)
    fields = None
    if keep_fields:
        fields = {k: _np(v) for k, v in dict(hrA=hrA, hcA=hcA, hrB=hrB, hcB=hcB, lnrowA=lnrowA, lncolA=lncolA, eZ_row=eZr, eZ_col=eZc, eW_row=eWr, eW_col=eWc,
                  theta_W_row=gW["theta"], theta_Z_row=gZ["theta"], par_row=gW["par"], perp_row=gW["perp"], rho_row=gW["rho"],
                  theta_W_col=gWc["theta"], theta_Z_col=gZc["theta"], par_col=gWc["par"], perp_col=gWc["perp"], rho_col=gWc["rho"],
                  eqstep_resid_Zrow=base_r["resid_fixed"], eqstep_resid_Zcol=base_c["resid_fixed"], theta_resid_row=th_r["resid_fixed"], theta_resid_col=th_c["resid_fixed"]).items()}
    return dict(closure=closure, L1=L1, L2=L2, L3=L3), fields


def to_jsonable(o):
    if isinstance(o, dict): return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if torch.is_tensor(o): return o.item() if o.numel() == 1 else o.tolist()
    return o
