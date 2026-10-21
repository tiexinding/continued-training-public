# 发布包：卡上运行队列的原始记录，仅供审计，不能直接运行（见本目录 README.md）；部分路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2），仍含卡上路径与清理命令
#!/bin/bash
# T1 换任务测试（10-05，老丁令"考虑换任务的计划，展开测试"）：第一任务改为 MBPP（0.5B 基座 pass@1 24%，有提升空间），
# 新任务收益 = MBPP pass@1，旧能力保持 = GSM8K 留出宽松正确率（基座 42.5%）与 Dolly/GSM8K 探针 NLL。
# lr ∈ {1e-5, 2e-5}，1 轮（874 题 / bs 8 ≈ 110 步），快照 10/25/50/75/100 + 终态，每点生成评估（GSM8K 200 + MBPP 100）。等 P1 结束后开跑。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4
cd $W/fork; mkdir -p $W/runs_t1 $W/status $W/logs $W/diag25/t1
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
until [ -f $W/status/DONE_p1 ]; do sleep 60; done
for LR in 1e-5 2e-5; do
  T=$W/runs_t1/lr$LR
  go t1_M_$LR python fork_continue.py segment --task mbpp --train $P/mbpp/train.jsonl --out $T/M --init base --switch fresh --lr $LR --epochs 1 --eval_every 25 --snap 10,25,50,75,100 $C || continue
  ST=""; for s in 10 25 50 75 100; do d=$(printf "%s/M/snaps/step%04d" $T $s); [ -f $d/weights.pt ] && ST="$ST s${s}_$LR=$d"; done
  ST="$ST final_$LR=$T/M/state"
  go t1_audit_$LR python $W/gen_audit.py --out $W/diag25/t1/lr$LR --max_new 400 --deterministic --mbpp --states $ST
  for s in 10 25 50 75 100; do rm -rf $(printf "%s/M/snaps/step%04d" $T $s); done
done
echo "T1 all done $(date +%T)"; touch $W/status/DONE_t1
