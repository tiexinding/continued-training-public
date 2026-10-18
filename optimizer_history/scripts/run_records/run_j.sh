# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# J 轮（文档 46 第 15 节，10-11）：一次顺序影响的验证干预。PG、留 v·t、125 步、lr 5e-6。
#   J0  --order_file 写出种子 17 原顺序，终点须逐位等于 E_vt（b20283ded2934fc1），否则 J_ABORT
#   J1  共享末段：PG_tail_s31、PG_tail_s47（最后 25 步 = 种子 17 的最后 25 批）
#   J2  末段衰减：PG_dec_s17/31/47（--decay_last 25）
# 每条：训练 → 成对场（--ab_only）→ HumanEval → 逐样本损失 → 删快照。磁盘约 5 G 空余，串行、同一时刻一份快照。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets
BG=$W/runs_e/B_g/state; J=$W/runs_j; D=$W/diag25/j; X=$W/diag25; O=$W/orders_j
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/fork; mkdir -p $J $D/segnll $W/status $W/logs
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status_j.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
PR4="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl --probe bio=$P/bio/probe.jsonl"

train() {   # train <name> <order_seed> <extra...>
  local n=$1 sd=$2; shift 2
  [ -f $W/status/DONE_j_train_$n ] || rm -rf $J/$n
  go j_train_$n python fork_continue.py fork --task gsm8k --train $P/gsm8k/train.jsonl --out $J/$n --init $BG \
     --epochs 1 --steps 125 --lr 5e-6 --order_seed $sd --full_eval 0,25,50,75,100,125 --dense "" --snap 125 \
     --tok_mode manual --deterministic $PR4 --m_op zero --v_op keep "$@"; }

readout() { # readout <name>
  local n=$1; local S=$J/$n/snaps/step0125
  [ -f $S/weights.pt ] || { [ -f $W/status/DONE_j_heval_$n ] && return 0; echo "no snapshot for $n"; return 1; }
  sha256sum $S/weights.pt | cut -c1-16 > $J/$n/endpoint_sha16.txt
  local T=$J/pairs_$n; mkdir -p $T/A $T/B; ln -sfn $BG $T/A/state; ln -sfn $S $T/B/state
  go j_pair_$n /root/miniconda3/bin/python $W/phase2_card.py --phase3 $T --out $D/pairs_$n --device cuda --ab_only
  go j_heval_$n python $W/drift_diag.py --stage heval --out $W/diag25/f1 --deterministic --tok_mode manual --tag j_$n --states "$n=$S"
  printf '[{"name":"%s","base":"B","swap_from":null,"groups":[]}]\n' $n > $D/cfg_$n.json
  go j_segnll_$n python state_mix_eval.py --stateA $BG --stateB $S --data $P --configs $D/cfg_$n.json --out $D/segnll --nll
  if [ -f $W/status/DONE_j_pair_$n ] && [ -f $W/status/DONE_j_heval_$n ] && [ -f $W/status/DONE_j_segnll_$n ]; then rm -rf $J/$n/snaps; fi
  df -h $W | tail -1; }

[ -f $W/status/J_ABORT ] && { echo "J_ABORT 存在，先查原因"; exit 1; }
# ---- J0 核对：顺序文件路径逐位复现 E_vt
if [ ! -f $W/status/DONE_j0_verify ]; then
  train PG_J0_s17 17 --order_file $O/order_J0_s17.json || { touch $W/status/J_ABORT; exit 1; }
  got=$(sha256sum $J/PG_J0_s17/snaps/step0125/weights.pt | cut -c1-16)
  echo -e "J0\t$got\tb20283ded2934fc1" >> $W/status/status_j.tsv
  [ "$got" = "b20283ded2934fc1" ] || { echo "J0 逐位核对失败 $got"; touch $W/status/J_ABORT; exit 1; }
  rm -rf $J/PG_J0_s17/snaps; touch $W/status/DONE_j0_verify
fi
# ---- J1 共享末段
for s in 31 47; do train PG_tail_s$s $s --order_file $O/order_J1_s$s.json && readout PG_tail_s$s; done
# ---- J2 末段衰减
for s in 17 31 47; do train PG_dec_s$s $s --decay_last 25 && readout PG_dec_s$s; done
# ---- 跨顺序通道一致性（J1 的种子 17 = 原运行 E_vt；基线同列）
F="PG_tail_s17=$X/e3/pairs_E_vt/A_B_fields.npz PG_dec_s17=$D/pairs_PG_dec_s17/A_B_fields.npz"
for s in 31 47; do F="$F PG_tail_s$s=$D/pairs_PG_tail_s$s/A_B_fields.npz PG_dec_s$s=$D/pairs_PG_dec_s$s/A_B_fields.npz PG_base_s$s=$X/h/pairs_PG_vt_s$s/A_B_fields.npz"; done
F="$F PG_base_s17=$X/e3/pairs_E_vt/A_B_fields.npz"
go j_channels python $W/cross_order_channels.py --fields $F --out $D/cross_order_channels_j.json
go j_pack bash -c "cd $W; tar czf /root/autodl-tmp/j_results.tgz diag25/j orders_j runs_j/*/log.jsonl runs_j/*/endpoint_sha16.txt \$(ls runs_j/*/records/probe_full_*.pt) \$(ls diag25/f1/heval_j_*.json) \$(ls logs/j_*.log) status/status_j.tsv; ls -la /root/autodl-tmp/j_results.tgz"
touch $W/status/DONE_j; echo "J done $(date +%T)"
