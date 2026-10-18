"""Mechanism figure v8 (2026-10-17): numbered badges 1-5 match Sections 3.1-3.5 (numbers in the caption only) and the bottom tier
has four panels, 3.2 row-column factors, 3.3 update allocation, 3.4 one-step response, 3.5 behavioral evaluation.
From v7.3."""

from pathlib import Path
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch, Circle

OUT = Path(os.environ.get('FIG_OUT', Path(__file__).resolve().parent))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'mathtext.fontset': 'dejavusans'})
INK = '#25343D'; GRAY = '#6E7B85'; LG = '#B9C4CC'; ST = '#5B6770'; PUR = '#7052A3'; TEAL = '#148B7B'
PC = {'keep_vt': '#2166AC', 'reset': '#4D4D4D', 'logrc': '#5AAE61', 'perm': '#E08214', 'same': '#BABABA'}
CM = 'Greys'
rng = np.random.default_rng(7)
W_IN = 6.5; XMAX = 180; YMAX = 122
KY = 1.0; FS = 1.0; Y0 = 78.0; DY = 0.0   # DY: upward shift of the top and middle tiers (gap to the bottom tier 4 -> 6)
def fy(y): return Y0 + (y - Y0) * KY + DY
def fh(h): return h * KY
fig, ax = plt.subplots(figsize=(W_IN, W_IN * YMAX / XMAX)); fig.subplots_adjust(0, 0, 1, 1); ax.axis('off')


def T(x, y, s, n=6.6, c=INK, ha='center', b=False, va='center', fam=None, z=6, rot=0):
    if FS != 1.0: n = max(5.7, n * FS)
    ax.text(x, fy(y), s, fontsize=n, color=c, ha=ha, va=va, weight='bold' if b else 'normal', linespacing=1.15,
            family=fam, zorder=z, rotation=rot)
def R(x, y, w, h, fc='white', ec=LG, lw=.7, z=1, ls='-'):
    ax.add_patch(Rectangle((x, fy(y)), w, fh(h), fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))
def RB(x, y, w, h, fc, ec, lw=.8, z=1, r=1.6):
    ax.add_patch(FancyBboxPatch((x, fy(y)), w, fh(h), boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z))
def A(p, q, c=GRAY, lw=.8, dash=False, ms=6, z=3):
    ax.add_patch(FancyArrowPatch((p[0], fy(p[1])), (q[0], fy(q[1])), arrowstyle='-|>', mutation_scale=ms, lw=lw, color=c,
                                 linestyle=(0, (3, 2)) if dash else '-', zorder=z, shrinkA=0, shrinkB=0))
def L(xs, ys, c=GRAY, lw=.8, dash=False, z=2):
    ax.plot(xs, [fy(v) for v in np.atleast_1d(ys)], color=c, lw=lw, ls=(0, (3, 2)) if dash else '-', zorder=z, solid_capstyle='butt')
def tile(x, y, w, h, M, vmin=None, vmax=None, ec=INK, lw=.5, z=2):
    ax.imshow(M, extent=(x, x + w, fy(y), fy(y + h)), cmap=CM, vmin=vmin, vmax=vmax, aspect='auto',
              interpolation='nearest', zorder=z); R(x, y, w, h, 'none', ec, lw, z=z + 1)
def mark(x, y, s, r=2.0):
    if FS != 1.0: r = 1.75
    ax.add_patch(Circle((x, fy(y)), r, fc=TEAL, ec='none', zorder=8)); T(x, y, s, 6.0, 'white', b=True, z=9)
def docs(x, y, c, s=1.0):
    for k in range(3): R(x + 1.1 * k * s, y - 1.1 * k * s, 7 * s, 9 * s, 'white', c, .6, z=2 + k)
    for j in range(3): L([x + 3.4 * s, x + 7.0 * s], [y + 3.4 * s - 2.0 * j * s] * 2, c, .55, z=6)
def vfield(r, c, s=1.0):
    return rng.normal(0, .35 * s, r)[:, None] + rng.normal(0, .7 * s, c)[None, :] + rng.normal(0, .12, (r, c))
def wbox(x, y, w, h, lab, light=False, stripe=None, k=None):
    """weight box; stripe='row'|'col' marks the intermediate-neuron axis with a dark band."""
    R(x, y, w, h, 'white', LG if light else INK, .6 if light else .7, z=4)
    if stripe == 'row': R(x + .35, y + h * .68, w - .7, h * .14, '#3A3A3A', 'none', z=5)
    if stripe == 'col': R(x + w * .58, y + .35, w * .13, h - .7, '#3A3A3A', 'none', z=5)
    T(x + w / 2 if stripe is None else (x + w * .32 if stripe == 'col' else x + w / 2),
      y + h / 2 if stripe != 'row' else y + h * .32, lab, 6.0, GRAY if light else INK, z=6)
def oplus(x, y, r=1.4):
    ax.add_patch(Circle((x, fy(y)), r * (.85 if FS != 1.0 else 1), fc='white', ec=ST, lw=.6, zorder=5)); T(x, y + .1, '+', 6.4, ST, z=6)


# ================= TIER 1: training loop with decoder (both stages), drawn smaller: it is the inner loop of the stages
KY = 0.78; FS = 0.95; DY = 2.0
RB(2, 78, 176, 51, '#FBFCFD', '#D3DADF', .7)
T(5, 125.4, 'TRAINING LOOP', 6.9, '#A3ADB5', ha='left', b=True)
yr = 111   # residual stream
docs(5, yr - 3, ST, .9); T(9, yr - 8.3, 'batch $B_k$', 6.2)
VR = [3, 7, 8]          # highlighted vocabulary rows (e.g. indentation tokens)
def vocab(x, y, w=5, h=14):
    M = rng.uniform(.1, .35, (12, 4)); M[VR] = .9; tile(x, y, w, h, M, 0, 1, INK, .6, z=4)
vocab(20, yr - 7); T(22.5, yr - 10.6, 'embed. $E$\n(tied)', 6.0)
A((13.6, yr), (19.6, yr), GRAY)
DX0, DX1 = 33, 137
R(DX0, 91.5, DX1 - DX0, 35, '#FFFFFF', ST, .7, z=1)
T(DX1 - 2, 123.6, 'decoder layer $\\ell$  ×24', 6.2, INK, ha='right')
A((25.4, yr), (DX0 + .1, yr), GRAY)
L([DX0, DX1], [yr, yr], ST, 1.1, z=2)
T(DX0 + 2.6, yr - 2.6, '$x_\\ell$', 6.4)
# attention branch (light: not analysed in this paper)
ya = 118.6
L([36, 36], [yr, ya], GRAY, .6); A((36, ya), (38.4, ya), GRAY, .6, ms=4)
wbox(38.5, ya - 2.3, 11.8, 4.6, 'RMSNorm', light=True)
for lab, yy in [('$W_q$', ya + 4.1), ('$W_k$', ya), ('$W_v$', ya - 4.1)]:
    A((50.4, ya), (53.0, yy), LG, .5, ms=3); wbox(53.1, yy - 1.8, 6.4, 3.6, lab, light=True)
    A((59.6, yy), (62.0, ya), LG, .5, ms=3)
wbox(62.1, ya - 2.3, 12, 4.6, 'attention', light=True)
A((74.2, ya), (76.4, ya), LG, .5, ms=3); wbox(76.5, ya - 1.8, 6.4, 3.6, '$W_o$', light=True)
L([83, 86, 86], [ya, ya, yr + 1.4], GRAY, .6); A((86, yr + 3), (86, yr + 1.45), GRAY, .6, ms=4)
oplus(86, yr)
# MLP branch (intermediate-neuron axis marked)
ym = 102.6
L([90, 90], [yr, ym], GRAY, .6); A((90, ym), (92.4, ym), GRAY, .6, ms=4)
wbox(92.5, ym - 2.3, 11.8, 4.6, 'RMSNorm', light=True)
for lab, yy in [('$W_{gate}$', ym + 3.4), ('$W_{up}$', ym - 3.4)]:
    A((104.4, ym), (106.9, yy), GRAY, .5, ms=3); wbox(107.0, yy - 2.7, 9.4, 5.4, lab, stripe='row')
for yy in (ym + 3.4, ym - 3.4): A((116.5, yy), (118.9, ym + (.9 if yy > ym else -.9)), GRAY, .5, ms=3)
ax.add_patch(Circle((119.9, fy(ym)), 1.15, fc='white', ec=ST, lw=.6, zorder=5)); T(119.9, ym, '⊙', 6.4, ST)
T(119.9, ym + 3.3, 'SiLU', 6.0, GRAY)
A((121.2, ym), (122.9, ym), GRAY, .5, ms=3)
R(123.0, ym - 4.0, 11.4, 8.0, 'white', INK, .7, z=4); R(132.0, ym - 3.65, 1.2, 7.3, '#3A3A3A', 'none', z=5)
T(127.6, ym, '$W_{down}$', 6.0, INK, z=6)
L([134.4, 135.4, 135.4], [ym, ym, yr - 1.4], GRAY, .6); A((135.4, yr - 3), (135.4, yr - 1.45), GRAY, .6, ms=4)
oplus(135.4, yr)
L([107.0, 107.0, 134.4, 134.4], [95.9, 95.0, 95.0, 95.9], TEAL, .7, z=5)
T(134.4, 93.3, 'neuron axis: gate/up rows, down columns', 6.0, TEAL, ha='right')
# final norm, LM head (tied), loss
A((DX1, yr), (138.9, yr), GRAY)
wbox(139.0, yr - 2.3, 7.0, 4.6, 'Norm', light=True)
A((146.1, yr), (148.4, yr), GRAY); vocab(148.5, yr - 7); T(151.0, yr - 10.6, 'head $E^\\top$\n(tied)', 6.0)
mark(151.0, yr + 9.6, '5')
A((153.6, yr), (156.2, yr), GRAY)
ax.add_patch(Circle((160, fy(yr)), 3.0, fc='white', ec=ST, lw=.8, zorder=3)); T(160, yr, '$\\mathcal{L}$', 8.6)
# optimizer loop (purple), right to left under the decoder
yl = 86.0
L([163.6, 172, 172], [yr, yr, yl + 4.4], PUR, .9); A((172, yl + 4.5), (172, yl + 3.9), PUR, .9)
tile(167, yl - 2.8, 10, 6.6, rng.normal(0, 1, (6, 8)), -2.4, 2.4, INK, .5)
T(170.5, yl - 4.6, 'gradient $g_k$', 6.0)
A((166.8, yl), (163.0, yl), PUR, .9)
R(140.4, yl - 4.2, 22.4, 8.4, '#FFFFFF', PUR, .8, z=3)
tile(141.4, yl - 3.4, 6.4, 6.8, rng.normal(0, 1, (6, 6)), -2.4, 2.4, PUR, .6, z=4)
tile(149.0, yl - 3.4, 6.4, 6.8, vfield(6, 6), -2, 2, PUR, .6, z=4)
R(156.6, yl - 2.0, 5.2, 4.0, 'white', PUR, .6, z=4); T(159.2, yl, '$t$', 6.6, PUR)
T(151.6, yl + 6.0, 'AdamW state  $m$, $v$, $t$', 6.0, PUR); mark(137.4, yl + 3.4, '2')
A((140.2, yl), (37.6, yl), PUR, .9)
L([36, 36], [yl, 91.4], PUR, .9); L([36, 37.8], [yl, yl], PUR, .9); A((36, 90.0), (36, 91.4), PUR, .9)
T(88, yl + 2.4, '$u=\\hat m/(\\sqrt{\\hat v}+\\epsilon)$,   $W\\leftarrow(1-\\eta\\lambda)W-\\eta u$', 6.2, PUR)
T(88, yl - 2.6, 'update all weights (incl. $E$)', 6.0, PUR)
mark(31.6, yl + 2.6, '3')
mark(129.0, yr + 4.6, '4')

KY = 1.0; FS = 1.0; DY = 2.0
for x, s, ha in [(10, 'loop on corpus A', 'left'), (170, 'loop on corpus B', 'right')]:
    A((x, 78), (x, 69.5), GRAY, dash=True); T(x + (2.2 if ha == 'left' else -2.2), 73.8, s, 6.0, GRAY, ha=ha)

# ================= TIER 2: stages and switch
RB(2, 36, 63, 33, '#F4F5F7', ST)
T(5, 65.2, 'STAGE A', 7.2, ST, ha='left', b=True); T(23, 65.2, 'learn task A', 6.4, GRAY, ha='left')
docs(6, 50.5, ST); T(10.5, 45.0, 'corpus A', 6.0, ST)
L([21, 58], [52, 52], ST, .9); A((57, 52), (61, 52), ST)
for x in (25, 39, 53):
    ax.add_patch(Circle((x, fy(52)), 1.1, fc=ST, ec='none', zorder=4)); tile(x - 4, 54.5, 8, 6.5, vfield(5, 7, .6), -2, 2, PUR, .6)
T(25, 48.6, '$S_0$', 6.6); T(39, 48.6, '$S_i$', 6.6); T(53, 48.6, '$S_T$', 6.6)
T(41, 44.2, 'historical $v$', 6.2, PUR)
mark(8, 39.6, '5'); T(11.5, 39.6, 'validate task A', 6.2, TEAL, ha='left')

RB(68, 36, 44, 33, '#F7F2FB', PUR)
T(90, 65.2, 'SWITCH', 7.2, PUR, b=True); mark(107.5, 65.2, '1')
T(90, 61.0, '$W$ shared:  $W^{(c)}_0=W_A$', 6.2)
blank = np.zeros((5, 7)); Vs = vfield(5, 7)
Vrc = Vs.mean(1, keepdims=True) + Vs.mean(0, keepdims=True) - Vs.mean()
Vp = Vs[rng.permutation(5)][:, rng.permutation(7)]
COLX = [76 + k * 8.8 for k in range(4)]
for x, nm, cc in zip(COLX, ['v,t', 'reset', 'log-RC', 'perm.'], [PC['keep_vt'], PC['reset'], PC['logrc'], PC['perm']]):
    T(x + 3.3, 56.9, nm, 6.0, cc, b=True)
for lab, y in [('$m$', 51.8), ('$v$', 45.4), ('$t$', 39.4)]: T(72, y, lab, 7.2, PUR)
for x in COLX: R(x, 49.4, 6.6, 4.8, 'white', PUR, .6); T(x + 3.3, 51.8, '0', 6.4, GRAY)
for x, V in zip(COLX, [Vs, blank, Vrc, Vp]): tile(x, 43.0, 6.6, 4.8, V, -2, 2, PUR, .6)
R(COLX[1], 43.0, 6.6, 4.8, 'white', PUR, .6, z=4); T(COLX[1] + 3.3, 45.4, '0', 6.4, GRAY, z=7)
for x, sym in zip(COLX, ['$T$', '$0$', '$T$', '$T$']): R(x, 37.5, 6.6, 3.8, 'white', PUR, .6, z=3); T(x + 3.3, 39.4, sym, 6.4, PUR)
A((65, 52), (68, 52), PUR); A((112, 52), (115, 52), PUR)

RB(115, 36, 63, 33, '#F4F5F7', ST)
T(118, 65.2, 'STAGE B', 7.2, ST, ha='left', b=True); T(136, 65.2, 'adapt to task B', 6.4, GRAY, ha='left')
docs(118, 50.5, ST); T(122.5, 45.0, 'corpus B', 6.0, ST)
x0, y0 = 136, 53
ax.add_patch(Circle((x0, fy(y0)), 1.2, fc=ST, ec='none', zorder=5)); T(x0, 49.2, '$S^{(c)}_0$', 6.4)
ends = [('keep_vt', 60), ('logrc', 56.8), ('perm', 53.6), ('reset', 49.5)]
for k, ye in ends:
    L([x0, 165], [y0, ye], PC[k], 1.1, z=3); ax.add_patch(Circle((165, fy(ye)), .9, fc=PC[k], ec='none', zorder=4))
L([x0, 165], [y0, 46.5], PC['same'], .9, dash=True); T(167, 46.5, 'A→A', 6.0, GRAY, ha='left')
T(167, 61, '$S^{(c)}_K$', 6.4, ha='left'); T(150, 43.4, 'condition paths (schematic)', 6.0, GRAY)
mark(120, 39.6, '5'); T(123.5, 39.6, 'validate B · retention on A', 6.2, TEAL, ha='left')

# ================= TIER 3: the measurements of Sections 3.2-3.5
DY = 0.0
PWS = [41.0, 41.0, 41.0, 46.6]; PX = [2.0]
for w in PWS[:-1]: PX.append(PX[-1] + w + 2.13)
for x0_, PW, n, h in zip(PX, PWS, ['2', '3', '4', '5'],
                     ['ROW–COLUMN FACTORS', 'UPDATE ALLOCATION', 'ONE-STEP RESPONSE', 'BEHAVIORAL EVALUATION']):
    RB(x0_, 2, PW, 30, 'white', TEAL, .9); mark(x0_ + 4, 28, n); T(x0_ + 7.4, 28, h, 6.2, TEAL, ha='left', b=True)
# (3.2) log v - mu = p_i + q_j + e_ij
x = PX[0]
Vb = vfield(7, 10); ra = Vb.mean(1, keepdims=True); cb = Vb.mean(0, keepdims=True); mu = Vb.mean()
tile(x + 3, 11, 9, 10, Vb - mu, -2, 2, PUR); T(x + 14, 16, '=', 8.4)
tile(x + 16, 11, 2.2, 10, ra - mu, -2, 2, PUR); T(x + 20.4, 16, '+', 8.4)
tile(x + 22.6, 14.3, 9, 3.2, cb - mu, -2, 2, PUR); T(x + 33.6, 16, '+', 8.4)
tile(x + 35.4, 11, 4.4, 10, Vb - ra - cb + mu, -2, 2, PUR)
T(x + 7.5, 23.4, '$\\log v-\\mu$', 6.2, PUR); T(x + 17.1, 23.4, '$p_i$', 6.2, PUR); T(x + 27.1, 23.4, '$q_j$', 6.2, PUR)
T(x + 37.6, 23.4, '$e_{ij}$', 6.2, PUR); T(x + 27.1, 11.6, 'neuron axis', 5.9, TEAL)
T(x + 20.5, 5.4, '$\\mu$ with both, one or no axis term', 6.0)
# (3.3) change energy of down columns vs new-task neurons
x = PX[1]
dW = rng.normal(0, .5, (7, 16)); hot = [2, 7, 12]; dW[:, hot] *= 3.2; cw = 23 / 16
tile(x + 3, 11.5, 23, 10, dW ** 2, 0, 6)
for j in hot: R(x + 3 + j * cw, 11.5, cw, 10, 'none', INK, 1.1, z=4)
for j in range(16): R(x + 3 + j * cw, 8.4, cw, 2.2, '#3A3A3A' if j in (2, 7, 12, 14) else '#E3E8EC', 'white', .3, z=3)
T(x + 14.5, 23.4, '$\\Delta W$, down columns', 5.9, GRAY); T(x + 14.5, 5.4, 'new-task neurons', 6.0, INK)
T(x + 34.5, 18.0, 'largest\nchanges', 6.0); A((x + 29.6, 18.0), (x + 26.6, 18.0), INK)
T(x + 34.5, 9.5, 'overlap', 6.0); A((x + 29.6, 9.5), (x + 26.6, 9.5), INK)
# (3.4) one fixed-state update: true v vs candidate X, local and end-to-end responses
x = PX[2]
G = rng.normal(0, 1, (6, 8))
tile(x + 3, 15.6, 7, 5.2, G / (1 + np.abs(vfield(6, 8, .6))), -2, 2, INK); T(x + 6.5, 23.0, '$\\delta W_T$', 6.0)
tile(x + 3, 5.6, 7, 5.2, G / (1 + np.abs(vfield(6, 8, .6))), -2, 2, INK); T(x + 6.5, 13.2, '$\\delta W_X$', 6.0)
L([x + 11.2, x + 11.2], [8.2, 18.2], GRAY, .6); A((x + 11.2, 13.2), (x + 13.8, 13.2), GRAY, ms=5)
T(x + 15, 19.2, 'local:', 5.9, GRAY, ha='left')
T(x + 15, 15.6, '$\\|\\Delta y_X-\\Delta y_T\\|\\,/\\,\\|\\Delta y_T\\|$', 6.2, ha='left')
T(x + 15, 11.4, 'end to end:', 5.9, GRAY, ha='left')
T(x + 15, 7.4, '$\\mathrm{KL}(p_T\\,\\|\\,p_X)$', 6.2, ha='left')
T(x + 22.0, 3.0, 'fixed $W$, $g$; one update per $v$', 6.0)
# (3.5) one condition, two data orders
x = PX[3]; PW = PWS[3]; C3 = PC['keep_vt']
for k, fc in enumerate([C3, 'white']): R(x + 4 + k * 5.4, 10, 3.0, 10, fc, C3, .9, z=3)
L([x + 2.6, x + 14], [10, 10], INK, .6); T(x + 8.2, 23.0, 'overlap', 6.0, GRAY); T(x + 8.2, 15, '≈', 8.4)
T(x + 5.5, 7.8, '17', 5.6, GRAY); T(x + 10.9, 7.8, '47', 5.6, GRAY)
for k, (fc, code) in enumerate([(C3, '    return x'), ('white', 'return x')]):
    y = 19.4 - k * 8.6; R(x + 16, y - 2.9, 24.4, 5.8, '#FAFBFC', C3, .9, z=3)
    T(x + 16.8, y, code, 6.0, INK, ha='left', fam='DejaVu Sans Mono'); R(x + 36.6, y - 1.2, 2.4, 2.4, fc, C3, .8, z=4)
T(x + 28.2, 15, '≠', 8.4); T(x + 28.2, 23.6, 'generated body', 5.9, GRAY)
T(x + PW / 2, 3.4, 'one condition, orders 17 and 47', 6.0, PUR)

ax.set(xlim=(0, XMAX), ylim=(0, YMAX))
for ext in ('png', 'svg', 'pdf'):
    fig.savefig(OUT / ('fig1_main_mechanism_v8_en.' + ext), dpi=450, facecolor='white')
plt.close(fig)
