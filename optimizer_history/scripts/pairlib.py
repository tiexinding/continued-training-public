#!/usr/bin/env python3
"""成对变化结构分析库（阶段 0/1）。文档 25 的计算实现。

约定（与 P5 一致，见 P5 §3.2 与 scripts/p5v2_twoaxis_bridge_v1.py::bothnorm_tracked）
  W = s · D_r · Z · D_c ,  D_r = diag(e^{h^r}),  D_c = diag(e^{h^c})
  Z 由交替行/列 RMS 归一化得到：每行、每列 RMS = 1（相对平坦度 < 1e-6）。
  规范（gauge）：s = RMS(W)（全局 RMS），mean(h^r) = mean(h^c)。
  行场 h^r 与列场 h^c 都是"减去共同水平后的对数尺度"；P5 的 H_row = sd(h^r)，H_col = sd(h^c)。

固定参照 A 的成对改动（评审 10-05 的恒等式）
  ΔZ^{(A)} = D_rA^{-1} (W_B − W_A) D_cA^{-1} / s_A ,   W_B = s_A D_rA (Z_A + ΔZ^{(A)}) D_cA   （恒等式，闭环核验）

两种几何口径（必须分开报）
  W 几何：原矢量空间的欧氏内积。第 i 行的转角 θ_i^W、平行分量 p_i、垂直分量 q_i 由 w_i^A 与 δw_i 直接算，
         也可在 Z 坐标用加权内积 ⟨x, y⟩_{D_cA²} = x D_cA² yᵀ 精确算出（本库两种都算并核验一致）。
  Z 几何：对 Z_A、ΔZ 用普通内积得到的角 θ_i^Z。它是另一个量（"尺度归一化坐标下的转角"），不与 θ_i^W 混用。

四层输出（每对检查点）
  L1 单状态：s、h^r、h^c、H_row、H_col、核心 Z 的元素分布描述（峭度；可选 P5 Weibull k 拟合）
  L2 尺度演化：Δs/s、Δh^r、Δh^c（配对）、corr/Spearman、sd(Δh)、ΔH
  L3 成对核心变化：ε=‖ΔW‖/‖W_A‖、整矩阵夹角 Θ、平行/垂直能量占比；行列场：e^r,e^c（ΔZ 能量）、ΔW 能量、ρ_i、θ_i^W、θ_i^Z、p_i、q_i；
     等步长基线：ln e_i^r 对 h_i^r 回归（等元素步长预测斜率 −2）与残差场；行列缩放可解释的改动能量占比（交替最小二乘）；
     ΔW/ΔZ 的奇异谱（MP 边界以上个数与能量）、ΔZ 能量落在 Z_A 前 r 奇异子空间内的占比（随机基线 r/n）；
     Z 坐标随机打乱零假设（能量场 CV 的零值 √(2/n)）
  L4 功能关联：另见 func 模块（需数据，阶段 1 可选）

所有计算 float64。
"""
from __future__ import annotations
import math
import numpy as np

EPS = 1e-300


# ---------------------------------------------------------------- 分解
def rms(W):
    return float(np.sqrt((np.asarray(W, np.float64) ** 2).mean()))


def decompose(W: np.ndarray, it: int = 500, tol: float = 1e-7):
    """返回 s_bal, hr, hc, Z，满足 W = s_bal·diag(e^hr)·Z·diag(e^hc)，Z 行列 RMS = 1，mean(hr)=mean(hc)=0。
    注意 s_bal ≠ RMS(W)：r_W² = s_bal²·mean[e^{2hr} Z² e^{2hc}]，两者相差 e^{2·lvl}（lvl 为平衡后 h 的共同均值，
    实测 q 矩阵约 −1.4%，窄场矩阵 <0.3%）。范数与范数变化一律用 rms(W)，分解与重构用 s_bal。"""
    W = np.asarray(W, dtype=np.float64)
    m, n = W.shape
    s = math.sqrt((W ** 2).mean())
    a = np.ones(m); b = np.ones(n); z = W / s
    for _ in range(it):
        r = np.sqrt((z ** 2).mean(1)); z = z / np.maximum(r, EPS)[:, None]; a = a * r
        c = np.sqrt((z ** 2).mean(0)); z = z / np.maximum(c, EPS)[None, :]; b = b * c
        rr = np.sqrt((z ** 2).mean(1)); cc = np.sqrt((z ** 2).mean(0))
        if rr.std() / rr.mean() < tol and cc.std() / cc.mean() < tol:
            break
    # gauge: mean(ln a) = mean(ln b)；此时 ∏ 不变，Z 不变
    la, lb = np.log(np.maximum(a, EPS)), np.log(np.maximum(b, EPS))
    g = 0.5 * (la.mean() - lb.mean())
    hr, hc = la - g, lb + g
    # 把残余的共同水平并入 s（理论上 ≈0）
    lvl = 0.5 * (hr.mean() + hc.mean())
    hr, hc = hr - lvl, hc - lvl
    s = s * math.exp(2 * lvl)
    rec = s * np.exp(hr)[:, None] * z * np.exp(hc)[None, :]
    err = np.abs(rec - W).max() / max(np.abs(W).max(), EPS)
    assert err < 1e-10, ("W = s D_r Z D_c violated", err)
    return s, hr, hc, z


def delta_z_ref(WA, WB, sA, hrA, hcA):
    """固定参照 A 坐标下的成对改动 ΔZ^{(A)}；并返回闭环重建误差。"""
    dW = np.asarray(WB, np.float64) - np.asarray(WA, np.float64)
    dZ = dW / (sA * np.exp(hrA)[:, None] * np.exp(hcA)[None, :])
    return dZ, dW


# ---------------------------------------------------------------- 几何
def row_geometry_W(WA, dW):
    """W 几何（直接法）：每行相对改动 ρ、平行分量 p、垂直分量 q（均相对 ‖w_i‖）、转角 θ^W（弧度）。"""
    na2 = (WA ** 2).sum(1); nd2 = (dW ** 2).sum(1); dot = (WA * dW).sum(1)
    rho = np.sqrt(nd2 / np.maximum(na2, EPS))
    p = dot / np.maximum(na2, EPS)
    q = np.sqrt(np.maximum(nd2 / np.maximum(na2, EPS) - p ** 2, 0.0))
    nb2 = na2 + 2 * dot + nd2
    cos = (na2 + dot) / np.sqrt(np.maximum(na2 * nb2, EPS))
    theta = np.arccos(np.clip(cos, -1, 1))
    return dict(rho=rho, par=p, perp=q, theta=theta, eA2=na2, eD2=nd2)


def row_geometry_Zweighted(ZA, dZ, hcA):
    """W 几何（Z 坐标加权法）：用加权内积 ⟨x,y⟩ = Σ_j x_j y_j e^{2 hc_j}。
    s_A、e^{hr_i} 在行内是公共因子，比值与角度不受影响，所以不出现。应与 row_geometry_W 逐位一致。"""
    w2 = np.exp(2 * hcA)[None, :]
    na2 = (ZA ** 2 * w2).sum(1); nd2 = (dZ ** 2 * w2).sum(1); dot = (ZA * dZ * w2).sum(1)
    rho = np.sqrt(nd2 / np.maximum(na2, EPS)); p = dot / np.maximum(na2, EPS)
    q = np.sqrt(np.maximum(nd2 / np.maximum(na2, EPS) - p ** 2, 0.0))
    nb2 = na2 + 2 * dot + nd2
    cos = (na2 + dot) / np.sqrt(np.maximum(na2 * nb2, EPS))
    return dict(rho=rho, par=p, perp=q, theta=np.arccos(np.clip(cos, -1, 1)))


def row_geometry_Zplain(ZA, dZ):
    """Z 几何（普通内积）：尺度归一化坐标下的转角 θ^Z 等。另一个量。"""
    return row_geometry_W(ZA, dZ)


def matrix_geometry(WA, dW):
    """整矩阵：ε、夹角 Θ、平行/垂直能量占比、范数相对变化。"""
    A2 = (WA ** 2).sum(); D2 = (dW ** 2).sum(); dot = (WA * dW).sum()
    eps = math.sqrt(D2 / A2)
    par = dot / A2                      # 平行分量（相对）
    par_energy = dot ** 2 / A2 / D2      # 改动能量中平行部分
    B2 = A2 + 2 * dot + D2
    cosT = (A2 + dot) / math.sqrt(A2 * B2)
    return dict(eps=eps, Theta=math.degrees(math.acos(max(-1, min(1, cosT)))), par_rel=par,
                par_energy_frac=par_energy, perp_energy_frac=1 - par_energy,
                norm_rel_change=math.sqrt(B2 / A2) - 1, cos_dW_WA=dot / math.sqrt(A2 * D2))


def length_vector_angles(WA, WB):
    """行/列长度向量夹角 Θ_r、Θ_c（整场夹角的严格下界），以及 P5 风格 h 向量（减中位数）夹角。"""
    out = {}
    for name, ax in (("row", 1), ("col", 0)):
        la = np.sqrt((WA ** 2).sum(ax)); lb = np.sqrt((WB ** 2).sum(ax))
        c = (la * lb).sum() / math.sqrt((la ** 2).sum() * (lb ** 2).sum())
        out[f"Theta_{name}_len_deg"] = math.degrees(math.acos(min(1, c)))
    return out


# ---------------------------------------------------------------- 行列缩放拟合（state_compare3 同法）
def fit_rowcol(Wf, Wt, iters=20):
    r = np.ones(Wf.shape[0]); c = np.ones(Wf.shape[1])
    for _ in range(iters):
        X = Wf * c[None, :]; r = (Wt * X).sum(1) / np.maximum((X ** 2).sum(1), EPS)
        X = Wf * r[:, None]; c = (Wt * X).sum(0) / np.maximum((X ** 2).sum(0), EPS)
    return r, c


def scaling_explained(WA, WB, sA=None, hrA=None, hcA=None, sB=None, hrB=None, hcB=None):
    """行列缩放能解释多少改动能量。两种口径：
    _ls：交替最小二乘拟合 W_B ≈ D_r W_A D_c（文档 24 / state_compare3 同法）。注意它不限制符号与收缩：
         当 W_B 与 W_A 近乎无关时（如翻号），最优拟合收缩到 0，会给出约 0.5 的"解释比例"，须与 _field 对照。
    _field：用分解测得的尺度演化本身，S' = (s_B/s_A)·diag(e^{Δh^r})·W_A·diag(e^{Δh^c})，即"只让尺度场按实测演化、核心不变"
         能实现的改动。scale_frac_field = ‖S'−W_A‖²/E_tot，res_frac_field = ‖W_B−S'‖²/E_tot（两者之和不必为 1）。"""
    dW = WB - WA; Etot = (dW ** 2).sum()

    def three(S, tag):
        U, V = S - WA, WB - S   # ΔW = U + V，一般不正交：subst + res + cross = 1
        u2, v2 = (U ** 2).sum() / Etot, (V ** 2).sum() / Etot
        return {f"subst_{tag}": u2, f"res_{tag}": v2, f"cross_{tag}": 1 - u2 - v2, f"resid_drop_{tag}": 1 - v2}

    r, c = fit_rowcol(WA, WB); a = (WB * WA).sum() / (WA ** 2).sum()
    out = dict(**three(r[:, None] * WA * c[None, :], "ls"), **three(a * WA, "global"))
    if sA is not None:
        # 实测场替代模型：尺度场按 A→B 实测演化、核心不变。整体幅度必须用 s_bal 比（纯缩放下恒等式精确成立），不能用 r_W 比
        Sf = (sB / sA) * np.exp(hrB - hrA)[:, None] * WA * np.exp(hcB - hcA)[None, :]
        out.update(three(Sf, "field"))
    return out


# ---------------------------------------------------------------- 能量场、等步长基线、零假设
def energy_fields(dZ, dW):
    return dict(eZ_row=(dZ ** 2).sum(1), eZ_col=(dZ ** 2).sum(0), eW_row=(dW ** 2).sum(1), eW_col=(dW ** 2).sum(0))


def equal_step_baseline(e_row, h_row, fixed_slope=-2.0):
    """等元素改动零模型：ΔW_ij = u·ξ_ij（ξ 同分布），则 e_i^r(ΔZ) ∝ e^{−2 h_i^r}·Σ_j e^{−2 h_j^c}。
    两种 R² 分开报：r2_fit = 自由拟合斜率与截距（样本内线性关联强度）；r2_fixed = 固定理论斜率、只拟合截距（零模型的预测精度，可为负）。
    用 W 能量时固定斜率为 0（r2_fixed 恒为 0，只看 r2_fit）；用行转角时固定斜率为 −1。"""
    y = np.log(np.maximum(e_row, EPS)); x = h_row
    X = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta; sst = ((y - y.mean()) ** 2).sum(); r2 = 1 - (res ** 2).sum() / sst
    pred_fixed = fixed_slope * x; pred_fixed = pred_fixed - pred_fixed.mean() + y.mean()
    r2_fixed = 1 - ((y - pred_fixed) ** 2).sum() / sst
    return dict(slope=float(beta[0]), r2_fit=float(r2), r2_fixed=float(r2_fixed), resid_fit=res, resid_fixed=y - pred_fixed)


def shuffle_null(dZ, dW, sA, hrA, hcA, rng, reps=5):
    """随机打乱零假设。两种坐标：
    W 坐标打乱（等步长零假设）：打乱 ΔW 的元素位置再换到 A 坐标，得到的 Z 能量场只含 e^{−2h} 基线；
        报实测 ΔZ 能量场 CV 与该零值之比，以及 W 能量场 CV 与其零值 √(2/n)（高斯近似）。
    Z 坐标打乱：打乱 ΔZ 元素位置；它检验"在归一化坐标下均匀"，与优化器零假设不同，仅作参考。"""
    m, n = dZ.shape; scale = sA * np.exp(hrA)[:, None] * np.exp(hcA)[None, :]
    cvZr_W, cvZc_W, cvWr, cvWc, cvZr_Z, cvZc_Z, r2fit_null, r2fix_null, last_pZ = [], [], [], [], [], [], [], [], None
    for _ in range(reps):
        pW = rng.permutation(dW.ravel()).reshape(m, n); pZ = pW / scale; last_pZ = pZ
        er, ec = (pZ ** 2).sum(1), (pZ ** 2).sum(0); cvZr_W.append(er.std() / er.mean()); cvZc_W.append(ec.std() / ec.mean())
        b = equal_step_baseline(er, hrA); r2fit_null.append(b["r2_fit"]); r2fix_null.append(b["r2_fixed"])
        er, ec = (pW ** 2).sum(1), (pW ** 2).sum(0); cvWr.append(er.std() / er.mean()); cvWc.append(ec.std() / ec.mean())
        p = rng.permutation(dZ.ravel()).reshape(m, n); er, ec = (p ** 2).sum(1), (p ** 2).sum(0)
        cvZr_Z.append(er.std() / er.mean()); cvZc_Z.append(ec.std() / ec.mean())
    er, ec = (dZ ** 2).sum(1), (dZ ** 2).sum(0); ewr, ewc = (dW ** 2).sum(1), (dW ** 2).sum(0)
    return dict(cvZ_row=er.std() / er.mean(), cvZ_col=ec.std() / ec.mean(),
                cvZ_row_null_Wshuffle=float(np.mean(cvZr_W)), cvZ_col_null_Wshuffle=float(np.mean(cvZc_W)),
                cvW_row=ewr.std() / ewr.mean(), cvW_col=ewc.std() / ewc.mean(),
                cvW_row_null=float(np.mean(cvWr)), cvW_col_null=float(np.mean(cvWc)), cvW_row_gauss=math.sqrt(2 / n), cvW_col_gauss=math.sqrt(2 / m),
                cvZ_row_null_Zshuffle=float(np.mean(cvZr_Z)), cvZ_col_null_Zshuffle=float(np.mean(cvZc_Z)),
                eqstep_r2fit_null=float(np.mean(r2fit_null)), eqstep_r2fixed_null=float(np.mean(r2fix_null)), _null_dZ=last_pZ)


# ---------------------------------------------------------------- 谱与子空间
def spectrum(D, rng=None):
    m, n = D.shape; Etot = (D ** 2).sum()
    sv = np.linalg.svd(D, compute_uv=False)
    edge = math.sqrt(Etot / (m * n)) * (math.sqrt(m) + math.sqrt(n))
    above = sv > edge
    out = dict(n_above_mp=int(above.sum()), energy_above_mp=float((sv[above] ** 2).sum() / (sv ** 2).sum()),
               erank=float((sv ** 2).sum() ** 2 / (sv ** 4).sum()), top1_frac=float(sv[0] ** 2 / (sv ** 2).sum()),
               top10_frac=float((sv[:10] ** 2).sum() / (sv ** 2).sum()))
    if rng is not None:
        R = rng.standard_normal((m, n)) * math.sqrt(Etot / (m * n)); rs = np.linalg.svd(R, compute_uv=False)
        out.update(rand_n_above_mp=int((rs > edge).sum()), rand_top10_frac=float((rs[:10] ** 2).sum() / (rs ** 2).sum()),
                   rand_erank=float((rs ** 2).sum() ** 2 / (rs ** 4).sum()))
    return out


def subspace_share(ZA, dZ, ranks=(8, 32, 128), null_dZ=None):
    """ΔZ 能量落在 Z_A 前 r 左/右奇异子空间内的占比。两种零值：
    _iso：各向同性随机方向 r/m（左）、r/n（右）；
    _shuf：W 坐标打乱后经同一尺度变换得到的 ΔZ（异方差零模型）在同一子空间内的占比（null_dZ 给定时）。
    只是结构量：不解读为"重写旧知识 / 新增知识"。"""
    U, S, Vt = np.linalg.svd(ZA, full_matrices=False); m, n = ZA.shape; E = (dZ ** 2).sum(); out = {}
    En = (null_dZ ** 2).sum() if null_dZ is not None else None
    for r in ranks:
        if r >= min(m, n): continue
        out[f"left_r{r}"] = float(((U[:, :r].T @ dZ) ** 2).sum() / E); out[f"left_r{r}_null_iso"] = r / m
        out[f"right_r{r}"] = float(((dZ @ Vt[:r].T) ** 2).sum() / E); out[f"right_r{r}_null_iso"] = r / n
        if null_dZ is not None:
            out[f"left_r{r}_null_shuf"] = float(((U[:, :r].T @ null_dZ) ** 2).sum() / En)
            out[f"right_r{r}_null_shuf"] = float(((null_dZ @ Vt[:r].T) ** 2).sum() / En)
    return out


def principal_overlap(ZA, ZB, ranks=(8, 32)):
    """A、B 自身主子空间的变化：overlap = ‖U_Aᵀ U_B‖²_F / r（1 = 完全重合；随机 ≈ r/m）。"""
    UA, _, VtA = np.linalg.svd(ZA, full_matrices=False); UB, _, VtB = np.linalg.svd(ZB, full_matrices=False); m, n = ZA.shape; out = {}
    for r in ranks:
        out[f"left_r{r}"] = float(((UA[:, :r].T @ UB[:, :r]) ** 2).sum() / r)
        out[f"right_r{r}"] = float(((VtA[:r] @ VtB[:r].T) ** 2).sum() / r); out[f"left_r{r}_null"] = r / m; out[f"right_r{r}_null"] = r / n
    return out


# ---------------------------------------------------------------- 核心形状描述
def core_shape(Z):
    x = Z.ravel(); x = x - x.mean()
    m2 = (x ** 2).mean(); return dict(kurt_excess=float((x ** 4).mean() / m2 ** 2 - 3), skew=float((x ** 3).mean() / m2 ** 1.5))


# ---------------------------------------------------------------- 四层汇总（单对矩阵）
def analyse_pair(WA, WB, rng=None, ranks=(8, 32, 128), with_spectrum=True, keep_fields=True):
    WA = np.asarray(WA, np.float64); WB = np.asarray(WB, np.float64)
    sA, hrA, hcA, ZA = decompose(WA); sB, hrB, hcB, ZB = decompose(WB); rA, rB = rms(WA), rms(WB)
    dZ, dW = delta_z_ref(WA, WB, sA, hrA, hcA)
    rec = sA * np.exp(hrA)[:, None] * (ZA + dZ) * np.exp(hcA)[None, :]
    closure = float(np.abs(rec - WB).max() / np.abs(WB).max())
    L1 = dict(s_bal_A=sA, s_bal_B=sB, rms_A=rA, rms_B=rB, s_bal_over_rms_A=sA / rA, H_row_A=hrA.std(), H_col_A=hcA.std(), H_row_B=hrB.std(), H_col_B=hcB.std(),
              coreA=core_shape(ZA), coreB=core_shape(ZB))
    dhr, dhc = hrB - hrA, hcB - hcA
    L2 = dict(drms_rel=rB / rA - 1, ds_bal_rel=sB / sA - 1, dH_row=hrB.std() - hrA.std(), dH_col=hcB.std() - hcA.std(),
              sd_dh_row=dhr.std(), sd_dh_col=dhc.std(), corr_h_row=np.corrcoef(hrA, hrB)[0, 1], corr_h_col=np.corrcoef(hcA, hcB)[0, 1],
              spearman_h_row=_spearman(hrA, hrB), spearman_h_col=_spearman(hcA, hcB),
              dh_vs_h_slope_row=np.polyfit(hrA, dhr, 1)[0], dh_vs_h_slope_col=np.polyfit(hcA, dhc, 1)[0])
    gW = row_geometry_W(WA, dW); gWz = row_geometry_Zweighted(ZA, dZ, hcA); gZ = row_geometry_Zplain(ZA, dZ)
    gWc = row_geometry_W(WA.T, dW.T); gZc = row_geometry_Zplain(ZA.T, dZ.T)
    agree = float(np.abs(gW["theta"] - gWz["theta"]).max())
    ef = energy_fields(dZ, dW)
    base_r = equal_step_baseline(ef["eZ_row"], hrA); base_c = equal_step_baseline(ef["eZ_col"], hcA)
    baseW_r = equal_step_baseline(ef["eW_row"], hrA, 0.0); baseW_c = equal_step_baseline(ef["eW_col"], hcA, 0.0)
    lnrowA = 0.5 * np.log(np.maximum(gW["eA2"], EPS)); lncolA = 0.5 * np.log(np.maximum(gWc["eA2"], EPS))   # 原始行/列 log 长度
    th_r = equal_step_baseline(gW["theta"], hrA, -1.0); th_c = equal_step_baseline(gWc["theta"], hcA, -1.0)
    th_r_raw = equal_step_baseline(gW["theta"], lnrowA, -1.0); th_c_raw = equal_step_baseline(gWc["theta"], lncolA, -1.0)
    # 逐行平行/垂直能量（不经整体内积抵消）
    par_rows = float((gW["par"] ** 2 * gW["eA2"]).sum() / gW["eD2"].sum()); par_cols = float((gWc["par"] ** 2 * gWc["eA2"]).sum() / gWc["eD2"].sum())
    L3 = dict(**matrix_geometry(WA, dW), **length_vector_angles(WA, WB), **scaling_explained(WA, WB, sA, hrA, hcA, sB, hrB, hcB),
              theta_W_deg_med=float(np.degrees(np.median(gW["theta"]))), theta_W_deg_p90=float(np.degrees(np.percentile(gW["theta"], 90))),
              theta_Z_deg_med=float(np.degrees(np.median(gZ["theta"]))),
              theta_Wcol_deg_med=float(np.degrees(np.median(gWc["theta"]))), theta_Zcol_deg_med=float(np.degrees(np.median(gZc["theta"]))),
              par_rel_row_med=float(np.median(gW["par"])), perp_rel_row_med=float(np.median(gW["perp"])),
              par_energy_rows=par_rows, par_energy_cols=par_cols, par_energy_rows_signed_frac=float(np.mean(gW["par"] > 0)),
              theta_W_vs_Zweighted_maxabs=agree, corr_theta_W_Z_row=float(np.corrcoef(gW["theta"], gZ["theta"])[0, 1]),
              eqstep_Zrow_slope=base_r["slope"], eqstep_Zrow_r2fit=base_r["r2_fit"], eqstep_Zrow_r2fixed=base_r["r2_fixed"],
              eqstep_Zcol_slope=base_c["slope"], eqstep_Zcol_r2fit=base_c["r2_fit"], eqstep_Zcol_r2fixed=base_c["r2_fixed"],
              eqstep_Wrow_slope=baseW_r["slope"], eqstep_Wrow_r2fit=baseW_r["r2_fit"], eqstep_Wcol_slope=baseW_c["slope"], eqstep_Wcol_r2fit=baseW_c["r2_fit"],
              theta_vs_h_slope_row=th_r["slope"], theta_vs_h_r2fit_row=th_r["r2_fit"], theta_vs_h_r2fixed_row=th_r["r2_fixed"],
              theta_vs_h_slope_col=th_c["slope"], theta_vs_h_r2fit_col=th_c["r2_fit"], theta_vs_h_r2fixed_col=th_c["r2_fixed"],
              theta_vs_lnrow_slope_row=th_r_raw["slope"], theta_vs_lnrow_r2fit_row=th_r_raw["r2_fit"], theta_vs_lnrow_r2fixed_row=th_r_raw["r2_fixed"],
              theta_vs_lncol_slope_col=th_c_raw["slope"], theta_vs_lncol_r2fit_col=th_c_raw["r2_fit"])
    null_dZ = None
    if rng is not None:
        sh = shuffle_null(dZ, dW, sA, hrA, hcA, rng); null_dZ = sh.pop("_null_dZ"); L3["shuffle"] = sh
    if with_spectrum:
        L3["spec_dW"] = spectrum(dW, rng); L3["spec_dZ"] = spectrum(dZ, rng)
        L3["subspace"] = subspace_share(ZA, dZ, ranks, null_dZ); L3["principal_overlap_AB"] = principal_overlap(ZA, ZB)
    fields = None
    if keep_fields:
        fields = dict(hrA=hrA, hcA=hcA, hrB=hrB, hcB=hcB, lnrowA=lnrowA, lncolA=lncolA, eZ_row=ef["eZ_row"], eZ_col=ef["eZ_col"], eW_row=ef["eW_row"], eW_col=ef["eW_col"],
                      theta_W_row=gW["theta"], theta_Z_row=gZ["theta"], par_row=gW["par"], perp_row=gW["perp"], rho_row=gW["rho"],
                      theta_W_col=gWc["theta"], theta_Z_col=gZc["theta"], par_col=gWc["par"], perp_col=gWc["perp"], rho_col=gWc["rho"],
                      eqstep_resid_Zrow=base_r["resid_fixed"], eqstep_resid_Zcol=base_c["resid_fixed"],
                      theta_resid_row=th_r["resid_fixed"], theta_resid_col=th_c["resid_fixed"])
    return dict(closure=closure, L1=L1, L2=L2, L3=L3), fields


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def _r2_lin(x, y):
    b = np.polyfit(x, y, 1); res = y - np.polyval(b, x); return float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum())


def to_jsonable(o):
    if isinstance(o, dict): return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o
