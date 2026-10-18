# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# I 轮 CPU 队列（与 run_i.sh 并行，nice 19、4 线程，不用 GPU）：
#   I3  h2p_energy --segments：H 轮 25 条新跑的逐步能量分段重排；
#   I4  cross_order_channels：36 格成对场的跨顺序通道一致性（标签与 run_h3.sh 相同）。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; D=$W/diag25/h; X=$W/diag25; O=$W/diag25/i
export PATH=$R/venv/bin:$PATH OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
cd $W; mkdir -p $O $W/status $W/logs
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  nice -n 19 "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status_cpu.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
RUNS=$(for d in $W/runs_h/*/records/upd_energy.pt; do n=$(basename $(dirname $(dirname $d))); echo -n "$n=${n%%_*} "; done)
go i3_segments python $W/h2p_energy.py --runs $RUNS --act MG=$D/neuron_act_M_hall.npz GM=$D/neuron_act_A_g_hall.npz PG=$D/neuron_act_B_g_hall.npz --out $O/h2p_energy_segments.json --segments
fl() { local F=""; for s in "$@"; do local p=${s#*=}; [ -f $p ] && F="$F $s" || echo "缺 $s" >&2; done; echo $F; }
new() { for sd in 31 47 17; do for c in vt vfac vperm 0; do [ -f $D/pairs_$1_${c}_s$sd/A_B_fields.npz ] && echo -n "$1_${c}_s$sd=$D/pairs_$1_${c}_s$sd/A_B_fields.npz "; done; done; }
MG_OLD="MG_vt_s17=$X/b2/pairs_Kv/A_B_fields.npz MG_0_s17=$X/t3/pairs/A_B_fields.npz MG_vt_s31=$X/c3/pairs_K_vt_s31/A_B_fields.npz MG_0_s31=$X/c3/pairs_K_0_s31/A_B_fields.npz"
GM_OLD="GM_vt_s17=$X/d3/pairs_R_vt/A_B_fields.npz GM_vperm_s17=$X/e3/pairs_R_vperm_med/A_B_fields.npz GM_0_s17=$X/d3/pairs_R_0/A_B_fields.npz"
PG_OLD="PG_vt_s17=$X/e3/pairs_E_vt/A_B_fields.npz PG_vfac_s17=$X/e3/pairs_E_vfac/A_B_fields.npz PG_vperm_s17=$X/e3/pairs_E_vperm/A_B_fields.npz PG_0_s17=$X/e3/pairs_E_0/A_B_fields.npz"
# 旧标签优先（与 run_h3 相同：PG_vt_s17 新跑是逐位复跑，用旧那份）；新跑中与旧标签重名的去掉
ALL=$(fl $MG_OLD $GM_OLD $PG_OLD $( (new MG; new GM; new PG) | tr ' ' '\n' | grep -v -E '^(MG_vt_s17|MG_0_s17|MG_vt_s31|MG_0_s31|GM_vt_s17|GM_vperm_s17|GM_0_s17|PG_vt_s17|PG_vfac_s17|PG_vperm_s17|PG_0_s17)=' | tr '\n' ' '))
echo "$ALL" | tr ' ' '\n' | grep -c = > $O/n_fields.txt
go i4_channels python $W/cross_order_channels.py --fields $ALL --out $O/cross_order_channels.json
go i_cpu_pack bash -c "cd $W; tar czf /root/autodl-tmp/i_cpu_results.tgz diag25/i/h2p_energy_segments.json diag25/i/cross_order_channels.json diag25/i/n_fields.txt logs/i3_*.log logs/i4_*.log status/status_cpu.tsv"
touch $W/status/DONE_i_cpu; echo "I CPU queue done $(date +%T)"
