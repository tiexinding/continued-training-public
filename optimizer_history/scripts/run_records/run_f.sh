# 发布包：卡上运行队列的原始记录，仅供审计，不能直接运行（见本目录 README.md）；部分路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2），仍含卡上路径与清理命令
#!/bin/bash
# F 轮（文档 38）：无监督维度漂移机制诊断。F1 分组 → F2 改动能量富集 → F3 首位置 logit → F4 撤回/移植干预 → F5 一步反事实 → F6 回传记录
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; E=$W/runs_e; M=$W/runs_t1/lr1e-5/M/state; AG=$W/runs_d/A_g/state; BG=$E/B_g/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/fork; mkdir -p $W/diag25/f1 $W/status $W/logs
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
C="--deterministic --tok_mode manual"
go f1_groups  python $W/drift_diag.py --stage groups $C
go f2_energy  python $W/drift_diag.py --stage energy $C --pairs "E_vt=$BG:$E/E_vt/snaps/step0125,E_vfac=$BG:$E/E_vfac/snaps/step0125,E_vperm=$BG:$E/E_vperm/snaps/step0125,E_0=$BG:$E/E_0/snaps/step0125,B_c=$BG:$E/B_c/state,P_vt=$M:$E/P_vt/snaps/step0125,P_0=$M:$E/P_0/snaps/step0125,R_vperm_med=$AG:$E/R_vperm_med/snaps/step0108,R_c=$AG:$W/runs_d/R_c/state"
go f3_logits  python $W/drift_diag.py --stage logits $C --states "B_g=$BG,E_vt=$E/E_vt/snaps/step0125,E_vfac=$E/E_vfac/snaps/step0125,E_vperm=$E/E_vperm/snaps/step0125,E_0=$E/E_0/snaps/step0125,B_c=$E/B_c/state,P_vt=$E/P_vt/snaps/step0125,P_0=$E/P_0/snaps/step0125,M=$M"
go f5_offline python $W/drift_diag.py --stage offline $C
go f4_interv  python $W/drift_diag.py --stage interv $C
touch $W/status/DONE_f
