# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# H 轮补算（10-09）：等 H2 结束后执行。
# (a) 全部格子（3 切换 × 4 条件 × 种子 17/31/47，新旧运行）用同一版 neuron_act（含 up 行原始能量）重算选择性重合与 ρ；
# (b) H2′：新跑的逐步更新能量累计排名 vs 净变化排名（h2p_energy.py）。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; D=$W/diag25/h; X=$W/diag25
M=$W/runs_t1/lr1e-5/M/state; AG=$W/runs_d/A_g/state; BG=$W/runs_e/B_g/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W
while [ ! -f $W/status/DONE_h2 ] && [ ! -f $W/status/H_ABORT ]; do sleep 60; done
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
fl() { local F=""; for s in "$@"; do local p=${s#*=}; [ -f $p ] && F="$F $s" || echo "缺 $s" >&2; done; echo $F; }
new() { for sd in 31 47 17; do for c in vt vfac vperm 0; do [ -f $D/pairs_$1_${c}_s$sd/A_B_fields.npz ] && echo -n "$1_${c}_s$sd=$D/pairs_$1_${c}_s$sd/A_B_fields.npz "; done; done; }
# 旧运行（种子 17 及 MG 已有种子 31）——标签统一为 切换_条件_s种子
MG_OLD="MG_vt_s17=$X/b2/pairs_Kv/A_B_fields.npz MG_0_s17=$X/t3/pairs/A_B_fields.npz MG_vt_s31=$X/c3/pairs_K_vt_s31/A_B_fields.npz MG_0_s31=$X/c3/pairs_K_0_s31/A_B_fields.npz"
GM_OLD="GM_vt_s17=$X/d3/pairs_R_vt/A_B_fields.npz GM_vperm_s17=$X/e3/pairs_R_vperm_med/A_B_fields.npz GM_0_s17=$X/d3/pairs_R_0/A_B_fields.npz"
PG_OLD="PG_vt_s17=$X/e3/pairs_E_vt/A_B_fields.npz PG_vfac_s17=$X/e3/pairs_E_vfac/A_B_fields.npz PG_vperm_s17=$X/e3/pairs_E_vperm/A_B_fields.npz PG_0_s17=$X/e3/pairs_E_0/A_B_fields.npz"
go h3_neuron_MG python $W/neuron_act.py --state $M  --name M_hall   --out $D --deterministic --fields $(fl $MG_OLD $(new MG))
go h3_neuron_GM python $W/neuron_act.py --state $AG --name A_g_hall --out $D --deterministic --fields $(fl $GM_OLD $(new GM))
# PG：新跑的 PG_vt_s17 是 E_vt 的逐位复跑，用旧标签那份即可，不重复
go h3_neuron_PG python $W/neuron_act.py --state $BG --name B_g_hall --out $D --deterministic --tasks gsm8k,mbpp,dolly_qa,bio --sel gsm8k,bio --fields $(fl $PG_OLD $(new PG | tr ' ' '\n' | grep -v '^PG_vt_s17=' | tr '\n' ' '))
RUNS=$(for d in $W/runs_h/*/records/upd_energy.pt; do n=$(basename $(dirname $(dirname $d))); echo -n "$n=${n%%_*} "; done)
go h3_energy python $W/h2p_energy.py --runs $RUNS --act MG=$D/neuron_act_M_hall.npz GM=$D/neuron_act_A_g_hall.npz PG=$D/neuron_act_B_g_hall.npz --out $D/h2p_energy.json
go h3_pack bash -c "cd $W; tar czf /root/autodl-tmp/h3_results.tgz diag25/h/neuron_act_*_hall.json diag25/h/h2p_energy.json logs/h3_*.log status/status.tsv; ls -la /root/autodl-tmp/h3_results.tgz"
touch $W/status/DONE_h3; echo "H3 done $(date +%T)"
