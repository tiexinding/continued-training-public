# 发布包：卡上运行队列的原始记录，仅供审计，不能直接运行（见本目录 README.md）；部分路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2），仍含卡上路径与清理命令
#!/bin/bash
# P1（10-05，老丁批）：低 lr 短训 + 多点生成评估。lr ∈ {1e-5, 5e-6, 2e-5}，GSM8K 1 轮（125 步），快照 10/25/50/75/100/125，
# 每个快照做 GSM8K 留出 200 题生成（400 token）+ MBPP 探针 100 题 pass@1；探针 NLL 每 25 步由训练脚本记录。每档完成后删中间快照（保留第 125 步）。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4
cd $W/fork; mkdir -p $W/runs_p1 $W/status $W/logs $W/diag25/p1
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
SNAPS="10,25,50,75,100,125"
for LR in 1e-5 5e-6 2e-5; do
  T=$W/runs_p1/lr$LR
  go p1_A_$LR python fork_continue.py segment --task gsm8k --train $P/gsm8k/train.jsonl --out $T/A --init base --switch fresh --lr $LR --epochs 1 --eval_every 25 --snap $SNAPS $C || continue
  ST=""; for s in 10 25 50 75 100 125; do d=$(printf "%s/A/snaps/step%04d" $T $s); [ -f $d/weights.pt ] && ST="$ST s${s}_$LR=$d"; done
  go p1_audit_$LR python $W/gen_audit.py --out $W/diag25/p1/lr$LR --max_new 400 --deterministic --mbpp --states $ST
  for s in 10 25 50 75 100; do rm -rf $(printf "%s/A/snaps/step%04d" $T $s); done
done
# 基座 MBPP/GSM8K 作同口径参照（已有 W0_max400，再补 MBPP）
go p1_audit_W0 python $W/gen_audit.py --out $W/diag25/p1/W0 --max_new 400 --deterministic --mbpp --states W0=base
echo "P1 all done $(date +%T)"; touch $W/status/DONE_p1
