# 发布包：卡上运行队列的原始记录，仅供审计，不能直接运行（见本目录 README.md）；部分路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2），仍含卡上路径与清理命令
#!/bin/bash
# 批次 3（10-06）：冻结 M→C 5e-6 候选的独立复验。
#  W3-1a 独立数据顺序：M→MBPP 再训 5e-6，顺序种子 31（原候选 23）；
#  W3-1b 独立评估集：HumanEval 164 题（/tmp/humaneval.parquet → jsonl），对 W0 / M / C(seed23) / C(seed31) / Gkeep 做 pass@1；
#  W3-2 旧能力：GSM8K 200 题生成 + MBPP 探针 100（gen_audit --mbpp）。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets; M=$W/runs_t1/lr1e-5/M/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4
cd $W/fork; mkdir -p $W/runs_b3 $W/status $W/logs $W/diag25/b3
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
go b3_C31 python fork_continue.py segment --task mbpp --train $P/mbpp/train.jsonl --out $W/runs_b3/C31 --init $M --switch keep --lr 5e-6 --epochs 1 --order_seed 31 --eval_every 25 $C
go b3_audit python $W/gen_audit.py --out $W/diag25/b3 --max_new 400 --deterministic --mbpp --states C31=$W/runs_b3/C31/state
go b3_humaneval python $W/humaneval_eval.py --out $W/diag25/b3/humaneval.json --states W0=base M=$M C23=$W/runs_t2/lr5e-6/C/state C31=$W/runs_b3/C31/state Gkeep=$W/runs_t2/lr5e-6/G/state Gfresh=$W/runs_t3/lr5e-6/G/state
rm -f $W/runs_b3/C31/state/optim.pt
echo "B3 all done $(date +%T)"; touch $W/status/DONE_b3
