# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# I5（文档 43 C 档，老丁 10-10 令"A+C 都算完"）：等 GPU 队列 run_i.sh 结束后，跑恢复实验的能量匹配随机对照（PG 置换，种子 17）。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/fork
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
while [ ! -f $W/status/DONE_i ] && [ ! -f $W/status/I_ABORT ]; do sleep 30; done
go i5_energy_match python $W/drift_diag.py --stage interv_energy --out $W/diag25/f1 --deterministic --tok_mode manual
go i5_pack bash -c "cd $W; tar czf /root/autodl-tmp/i5_results.tgz diag25/f1/intervene_energy.json logs/i5_*.log status/status.tsv; ls -la /root/autodl-tmp/i5_results.tgz"
touch $W/status/DONE_i5; echo "I5 done $(date +%T)"
