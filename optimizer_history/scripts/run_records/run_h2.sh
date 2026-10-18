# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# H2（文档 41 §3）：校准核对 —— 逐位重算 c、独立批次上的实际匹配误差、log v 行列模型 R² 口径。等 run_h.sh 结束（DONE_h）后串行执行（显存不与训练并用）。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS
M=$W/runs_t1/lr1e-5/M/state; AG=$W/runs_d/A_g/state; BG=$W/runs_e/B_g/state; D=$W/diag25/h/calib
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W; mkdir -p $D
while [ ! -f $W/status/DONE_h ] && [ ! -f $W/status/H_ABORT ]; do sleep 60; done
[ -f $W/status/H_ABORT ] && { echo "H_ABORT，不跑 H2"; exit 1; }
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
ck() { local n=$1 init=$2 task=$3 vop=$4 mc=$5; shift 5; go h2_$n python $W/calib_check.py --init $init --task $task --v_op $vop --order_seed 17 --match_c $mc --out $D/$n.json "$@"; }
# GM（A_g → MBPP）：RMS 校准两条（D 轮）、置换·中位数（E 轮）、行列近似·中位数（H3 新跑）
ck GM_vfac_rms   $AG mbpp  ladder:rowcol:matched     $W/runs_d/R_vfac/records/match_c.pt --r2
ck GM_vperm_rms  $AG mbpp  ladder:cperm:matched      $W/runs_d/R_vperm/records/match_c.pt
ck GM_vperm_med  $AG mbpp  ladder:cperm:matched_med  $W/runs_e/R_vperm_med/records/match_c.pt
ck GM_vfac_med   $AG mbpp  ladder:rowcol:matched_med $W/runs_h/GM_vfac_s17/records/match_c.pt
# PG（B_g → GSM8K）：两条中位数（E 轮）
ck PG_vfac_med   $BG gsm8k ladder:rowcol:matched_med $W/runs_e/E_vfac/records/match_c.pt --r2
ck PG_vperm_med  $BG gsm8k ladder:cperm:matched_med  $W/runs_e/E_vperm/records/match_c.pt
# MG（M → GSM8K）：两条中位数（H3 新跑）；--r2 给出 93% 的口径（MBPP 起点）
ck MG_vfac_med   $M  gsm8k ladder:rowcol:matched_med $W/runs_h/MG_vfac_s17/records/match_c.pt --r2
ck MG_vperm_med  $M  gsm8k ladder:cperm:matched_med  $W/runs_h/MG_vperm_s17/records/match_c.pt
go h2_pack bash -c "cd $W; tar czf /root/autodl-tmp/h2_results.tgz diag25/h/calib logs/h2_*.log status/status.tsv; ls -la /root/autodl-tmp/h2_results.tgz"
touch $W/status/DONE_h2; echo "H2 done $(date +%T)"
