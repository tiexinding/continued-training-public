# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# 批次 2（10-06）：从 M 出发、GSM8K、lr 5e-6、1 轮 125 步、顺序种子 17（与 T2 G keep / T3 G fresh 同批次同逐步 lr）的状态成分隔离：
#   Kv     : m 清零、v 与 t 保留      （fork --m_op zero --v_op keep）
#   bridge : m、v 清零、t 保留        （fork --m_op zero --v_op reset）
#   Cfresh : 同任务 MBPP 再训、清零整套状态（segment --switch fresh，顺序种子 23，与 T2 C keep 匹配）
# 逐步廉价记录：dense 步 1,2,5,10,25,50,75,100,125 存 scalars（lr/梯度范数/裁剪/更新 RMS 等）与 fields（第①层行列场与通道能量）；快照 10/50/终态做生成评估。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets; M=$W/runs_t1/lr1e-5/M/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4
cd $W/fork; mkdir -p $W/runs_b2 $W/status $W/logs $W/diag25/b2
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
DENSE="1,2,5,10,25,50,75,100,125"
go b2_Kv     python fork_continue.py fork --task gsm8k --train $P/gsm8k/train.jsonl --out $W/runs_b2/Kv     --init $M --m_op zero --v_op keep  --epochs 1 --steps 125 --order_seed 17 --lr 5e-6 --dense $DENSE --fields --full_eval 0,10,25,50,75,100,125 --snap 10,50,125 $C
go b2_bridge python fork_continue.py fork --task gsm8k --train $P/gsm8k/train.jsonl --out $W/runs_b2/bridge --init $M --m_op zero --v_op reset --epochs 1 --steps 125 --order_seed 17 --lr 5e-6 --dense $DENSE --fields --full_eval 0,10,25,50,75,100,125 --snap 10,50,125 $C
go b2_Cfresh python fork_continue.py segment --task mbpp --train $P/mbpp/train.jsonl --out $W/runs_b2/Cfresh --init $M --switch fresh --lr 5e-6 --epochs 1 --order_seed 23 --eval_every 25 --snap 10,50 $C
for A in Kv bridge; do
  ST="${A}10=$W/runs_b2/$A/snaps/step0010 ${A}50=$W/runs_b2/$A/snaps/step0050 ${A}final=$W/runs_b2/$A/snaps/step0125"
  go b2_audit_$A python $W/gen_audit.py --out $W/diag25/b2 --max_new 400 --deterministic --mbpp --states $ST
done
go b2_audit_Cfresh python $W/gen_audit.py --out $W/diag25/b2 --max_new 400 --deterministic --mbpp --states Cfresh10=$W/runs_b2/Cfresh/snaps/step0010 Cfresh50=$W/runs_b2/Cfresh/snaps/step0050 Cfreshfinal=$W/runs_b2/Cfresh/state
# 四层分析：A=M，B=臂终态，C=T2 的 C keep（同任务参照）
for A in Kv bridge Cfresh; do
  T=$W/runs_b2/pairs_$A; mkdir -p $T/A $T/B $T/C; ln -sfn $M $T/A/state
  if [ "$A" = "Cfresh" ]; then ln -sfn $W/runs_b2/Cfresh/state $T/B/state; else ln -sfn $W/runs_b2/$A/snaps/step0125 $T/B/state; fi
  ln -sfn $W/runs_t2/lr5e-6/C/state $T/C/state
  go b2_pair_$A /root/miniconda3/bin/python $W/phase2_card.py --phase3 $T --out $W/diag25/b2/pairs_$A --device cuda
done
echo "B2 all done $(date +%T)"; touch $W/status/DONE_b2
