# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# T3（10-05，老丁批）：检验"优化器状态失配"假说。M→GSM8K，lr 5e-6，1 轮 125 步，--switch fresh（重置 AdamW 状态），
# 与 T2 的 G 5e-6（keep）同起点同数据顺序只差状态处理。终态生成评估 + 四层分析（A=M, B=G_fresh, C=T2 的 C 5e-6）。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets; M=$W/runs_t1/lr1e-5/M/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4
cd $W/fork; mkdir -p $W/runs_t3/lr5e-6 $W/status $W/logs $W/diag25/t3
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
T=$W/runs_t3/lr5e-6
go t3_Gfresh python fork_continue.py segment --task gsm8k --train $P/gsm8k/train.jsonl --out $T/G --init $M --switch fresh --lr 5e-6 --epochs 1 --eval_every 25 --snap 10,50 $C
ST=""; for s in 10 50; do d=$(printf "%s/G/snaps/step%04d" $T $s); [ -f $d/weights.pt ] && ST="$ST Gfresh${s}=$d"; done; ST="$ST Gfreshfinal=$T/G/state"
go t3_audit python $W/gen_audit.py --out $W/diag25/t3 --max_new 400 --deterministic --mbpp --states $ST
mkdir -p $T/A $T/B $T/C; ln -sfn $M $T/A/state; ln -sfn $T/G/state $T/B/state; ln -sfn $W/runs_t2/lr5e-6/C/state $T/C/state
go t3_pair /root/miniconda3/bin/python $W/phase2_card.py --phase3 $T --out $W/diag25/t3/pairs --device cuda
rm -rf $T/G/snaps; rm -f $T/G/state/optim.pt
echo "T3 all done $(date +%T)"; touch $W/status/DONE_t3
