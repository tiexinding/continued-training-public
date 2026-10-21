# 发布包：卡上运行队列的原始记录，仅供审计，不能直接运行（见本目录 README.md）；部分路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2），仍含卡上路径与清理命令
#!/bin/bash
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; E=$W/runs_e; BG=$E/B_g/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/fork; mkdir -p $W/diag25/g1
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
ST="E_vt=$E/E_vt/snaps/step0125,E_vfac=$E/E_vfac/snaps/step0125,E_vperm=$E/E_vperm/snaps/step0125,E_0=$E/E_0/snaps/step0125,E_v=$E/E_v/snaps/step0125,E_vfac_t0=$E/E_vfac_t0/snaps/step0125,E_vperm_t0=$E/E_vperm_t0/snaps/step0125"
go g0_smoke python $W/mech_diag.py --stage onestep --K 2 --n_prompts 16 --out $W/diag25/g1/smoke --deterministic
go g1_decomp python $W/mech_diag.py --stage decomp --states "$ST" --deterministic
go g2_untie  python $W/mech_diag.py --stage untie  --states "$ST" --deterministic
go g3_onestep python $W/mech_diag.py --stage onestep --K 25 --deterministic
go g4_fin bash -c "cd $W; tar czf /root/autodl-tmp/g_results.tgz diag25/g1 logs/g*.log; ls -la /root/autodl-tmp/g_results.tgz"
touch $W/status/DONE_g
