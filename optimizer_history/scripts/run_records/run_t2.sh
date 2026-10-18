# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# T2（10-05，老丁令"从 M 出发做 M→GSM8K"）：以 T1 的 M（基座→MBPP，lr 1e-5，108 步，含优化器状态）为同起点，
#   G_lr = M→GSM8K 1 轮 125 步（keep 状态），lr 5e-6 / 1e-5；  C_lr = M→MBPP 再 1 轮（keep，同任务对照，顺序种子 23），lr 5e-6 / 1e-5。
#   每段快照 10/25/50/75/100 + 终态，各做 GSM8K 留出 200 题生成 + MBPP 探针 100 题 pass@1；探针 NLL 每 25 步。
#   训练完成后 GPU 后端四层分析：W0→M、M→G、M→C（每档），残差通道对应与读入—写出对应。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets; M=$W/runs_t1/lr1e-5/M/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4
cd $W/fork; mkdir -p $W/runs_t2 $W/status $W/logs $W/diag25/t2
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
SN="10,25,50,75,100"
for LR in 5e-6 1e-5; do
  T=$W/runs_t2/lr$LR; mkdir -p $T
  go t2_G_$LR python fork_continue.py segment --task gsm8k --train $P/gsm8k/train.jsonl --out $T/G --init $M --switch keep --lr $LR --epochs 1 --eval_every 25 --snap $SN $C
  ST=""; for s in 10 25 50 75 100; do d=$(printf "%s/G/snaps/step%04d" $T $s); [ -f $d/weights.pt ] && ST="$ST G${s}_$LR=$d"; done; ST="$ST Gfinal_$LR=$T/G/state"
  go t2_auditG_$LR python $W/gen_audit.py --out $W/diag25/t2/lr$LR --max_new 400 --deterministic --mbpp --states $ST
  for s in 10 25 50 75 100; do rm -rf $(printf "%s/G/snaps/step%04d" $T $s); done
  go t2_C_$LR python fork_continue.py segment --task mbpp --train $P/mbpp/train.jsonl --out $T/C --init $M --switch keep --lr $LR --epochs 1 --eval_every 25 --order_seed 23 --snap $SN $C
  ST=""; for s in 10 25 50 75 100; do d=$(printf "%s/C/snaps/step%04d" $T $s); [ -f $d/weights.pt ] && ST="$ST C${s}_$LR=$d"; done; ST="$ST Cfinal_$LR=$T/C/state"
  go t2_auditC_$LR python $W/gen_audit.py --out $W/diag25/t2/lr$LR --max_new 400 --deterministic --mbpp --states $ST
  for s in 10 25 50 75 100; do rm -rf $(printf "%s/C/snaps/step%04d" $T $s); done
  # 四层分析布局：A=M，B=G，C=C
  mkdir -p $T/A; ln -sfn $M $T/A/state; ln -sfn $T/G/state $T/B_state_link
  mkdir -p $T/B; ln -sfn $T/G/state $T/B/state
  go t2_pair_$LR /root/miniconda3/bin/python $W/phase2_card.py --phase3 $T --out $W/diag25/t2/lr$LR/pairs --device cuda
  rm -f $T/G/state/optim.pt $T/C/state/optim.pt
done
echo "T2 all done $(date +%T)"; touch $W/status/DONE_t2
