# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# 新卡状态重建（10-06 晚，原卡被占）：按原配方确定性重建 M / K_mvt / C_mvt / K_∅ 与旧臂 K_vt / K_t / C_∅ / C31，再做 t2/t3 成对分析与 neuron_act_M；最后核对指纹。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets; M=$W/runs_t1/lr1e-5/M/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/fork; mkdir -p $W/status $W/logs $W/diag25/t2/lr5e-6 $W/diag25/t3 $W/diag25/b4 $W/runs_t1/lr1e-5 $W/runs_t2/lr5e-6 $W/runs_t3/lr5e-6 $W/runs_b2 $W/runs_b3
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
go r_W0 python - <<PYEOF
import sys, torch; sys.path.insert(0, "$OH_TRAIN"); import forklib as fl
from fork_continue import load_model
class A: model="Qwen/Qwen2.5-0.5B-Instruct"; tokenizer=None; device="cpu"
m, tok = load_model(A()); import os; os.makedirs("$OH_RUNS/runs/W0", exist_ok=True)
torch.save({n: p.detach().cpu() for n, p in fl.named_params(m)}, "$OH_RUNS/runs/W0/weights.pt"); print("W0 saved")
PYEOF
go r_M      python fork_continue.py segment --task mbpp  --train $P/mbpp/train.jsonl  --out $W/runs_t1/lr1e-5/M --init base --switch fresh --lr 1e-5 --epochs 1 --eval_every 25 $C
go r_Kmvt   python fork_continue.py segment --task gsm8k --train $P/gsm8k/train.jsonl --out $W/runs_t2/lr5e-6/G --init $M --switch keep  --lr 5e-6 --epochs 1 --eval_every 25 $C
go r_Cmvt   python fork_continue.py segment --task mbpp  --train $P/mbpp/train.jsonl  --out $W/runs_t2/lr5e-6/C --init $M --switch keep  --lr 5e-6 --epochs 1 --eval_every 25 --order_seed 23 $C
go r_K0     python fork_continue.py segment --task gsm8k --train $P/gsm8k/train.jsonl --out $W/runs_t3/lr5e-6/G --init $M --switch fresh --lr 5e-6 --epochs 1 --eval_every 25 $C
DENSE="1,2,5,10,25,50,75,100,125"
go r_Kvt    python fork_continue.py fork --task gsm8k --train $P/gsm8k/train.jsonl --out $W/runs_b2/Kv     --init $M --m_op zero --v_op keep  --epochs 1 --steps 125 --order_seed 17 --lr 5e-6 --dense $DENSE --fields --full_eval 0,10,25,50,75,100,125 --snap 125 $C
go r_Kt     python fork_continue.py fork --task gsm8k --train $P/gsm8k/train.jsonl --out $W/runs_b2/bridge --init $M --m_op zero --v_op reset --epochs 1 --steps 125 --order_seed 17 --lr 5e-6 --dense $DENSE --fields --full_eval 0,10,25,50,75,100,125 --snap 125 $C
go r_C0     python fork_continue.py segment --task mbpp --train $P/mbpp/train.jsonl --out $W/runs_b2/Cfresh --init $M --switch fresh --lr 5e-6 --epochs 1 --order_seed 23 --eval_every 25 $C
go r_C31    python fork_continue.py segment --task mbpp --train $P/mbpp/train.jsonl --out $W/runs_b3/C31   --init $M --switch keep  --lr 5e-6 --epochs 1 --order_seed 31 --eval_every 25 $C
rm -f $W/runs_b2/Cfresh/state/optim.pt $W/runs_b3/C31/state/optim.pt
# 成对分析（A=M，B=G，C=C）与 T3
T=$W/runs_t2/lr5e-6; mkdir -p $T/A $T/B; ln -sfn $M $T/A/state; ln -sfn $T/G/state $T/B/state
go r_pair_t2 /root/miniconda3/bin/python $W/phase2_card.py --phase3 $T --out $W/diag25/t2/lr5e-6/pairs --device cuda
T=$W/runs_t3/lr5e-6; mkdir -p $T/A $T/B $T/C; ln -sfn $M $T/A/state; ln -sfn $W/runs_t3/lr5e-6/G/state $T/B/state; ln -sfn $W/runs_t2/lr5e-6/C/state $T/C/state
go r_pair_t3 /root/miniconda3/bin/python $W/phase2_card.py --phase3 $T --out $W/diag25/t3/pairs --device cuda
go r_act python $W/neuron_act.py --state $M --name M --deterministic --fields Gkeep=$W/diag25/t2/lr5e-6/pairs/A_B_fields.npz Gfresh=$W/diag25/t3/pairs/A_B_fields.npz
# 指纹核对（原卡 sha256 前 16 位：M 5d9825b68551462d / K_mvt 6d86dfaaf4c7e71c / C_mvt a6693161b3fc3fc3 / K_∅ b3bb41eb07bcf290；K_vt 256bf49c… / K_t 3825ece6… / C_∅ 14a4ef84… / C31 eb72d3db…）
for f in $M/weights.pt $W/runs_t2/lr5e-6/G/state/weights.pt $W/runs_t2/lr5e-6/C/state/weights.pt $W/runs_t3/lr5e-6/G/state/weights.pt $W/runs_b2/Kv/snaps/step0125/weights.pt $W/runs_b2/bridge/snaps/step0125/weights.pt $W/runs_b2/Cfresh/state/weights.pt $W/runs_b3/C31/state/weights.pt; do sha256sum $f; done > $W/status/fingerprints_rebuild.txt 2>&1; cat $W/status/fingerprints_rebuild.txt
echo "R all done $(date +%T)"; touch $W/status/DONE_r
