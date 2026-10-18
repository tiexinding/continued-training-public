# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# 阶段 3（10-05，老丁批两档 lr）：保能力训练设置下的同起点分叉 + 生成能力核查 + 四层端点分析。
# 每档 lr ∈ {2e-5, 5e-5}：A_lr = base→GSM8K 216 步（fresh）；B_lr = A_lr→MBPP 216 步（keep）；C_lr = A_lr→GSM8K 再 216 步（keep，同起点对照）。
# 其余设置与文档 22/24 的 Ren 配方相同（全参、恒定 lr、wd .01、clip 1、2 轮 = 216 步、顺序种子 17），只改 lr。
# 评估：state_mix_eval --nll --gen 于 W0 / A_lr / B_lr / C_lr（生成正确率 = 能力核查，基座 GSM8K 宽松 42.5%、MBPP 24%）。
# 分析：phase2_card.py 同口径（pairlib）→ diag25/phase3_<lr>/。
# 可重入：status/DONE_<名> 存在即跳过。先等阶段 2 的 CPU 分析结束（diag25/nearzero_turnover.json 出现）再开跑，避免 CPU 配额争抢。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4
cd $W/fork; mkdir -p $W/runs3 $W/status $W/logs $W/diag25
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }

echo "waiting for phase2 (diag25/nearzero_turnover.json) $(date +%T)"
while [ ! -f $W/diag25/nearzero_turnover.json ]; do sleep 60; done
echo "phase2 done, start phase3 $(date +%T)"

for LR in 2e-5 5e-5; do
  T=$W/runs3/lr$LR; mkdir -p $T
  go p3_A_$LR python fork_continue.py segment --task gsm8k --train $P/gsm8k/train.jsonl --out $T/A --init base    --switch fresh --lr $LR --eval_every 25 $C || continue
  go p3_B_$LR python fork_continue.py segment --task mbpp  --train $P/mbpp/train.jsonl  --out $T/B --init $T/A/state --switch keep --lr $LR --eval_every 25 $C
  go p3_C_$LR python fork_continue.py segment --task gsm8k --train $P/gsm8k/train.jsonl --out $T/C --init $T/A/state --switch keep --lr $LR --eval_every 25 $C
  # 生成能力核查 + 探针 NLL（W0 / A / B；C 用第二次调用，stateB=C）
  cat > $T/cfg_AB.json <<EOF
[{"name":"W0","base":"W0","swap_from":null,"groups":[]},{"name":"A","base":"A","swap_from":null,"groups":[]},{"name":"B","base":"B","swap_from":null,"groups":[]}]
EOF
  cat > $T/cfg_C.json <<EOF
[{"name":"C","base":"B","swap_from":null,"groups":[]}]
EOF
  go p3_eval_AB_$LR python state_mix_eval.py --stateA $T/A/state --stateB $T/B/state --data $P --configs $T/cfg_AB.json --out $W/diag25/phase3_lr$LR/eval --nll --gen
  go p3_eval_C_$LR  python state_mix_eval.py --stateA $T/A/state --stateB $T/C/state --data $P --configs $T/cfg_C.json  --out $W/diag25/phase3_lr$LR/eval --nll --gen
  # 四层端点分析（CPU，8 线程）：W0→A、A→B、A→C
  go p3_pair_$LR env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PH3_LR=$LR /root/miniconda3/bin/python $W/phase2_card.py --phase3 $T --out $W/diag25/phase3_lr$LR
  # 释放优化器状态（权重保留）
  rm -f $T/A/state/optim.pt $T/B/state/optim.pt $T/C/state/optim.pt
done
echo "phase3 all done $(date +%T)"; touch $W/status/DONE_phase3
