# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# K 轮（文档 47 v2）。分阶段：bash run_k.sh 1 → 冒烟 + legacy_check + 计时试跑（fp32 / fp64 各 1 批 8 题）；查看后再 bash run_k.sh 2（MG 主分析）、bash run_k.sh 3（PG 复验，预授权）。
set +e
STAGE=$1; R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets; K=$W/diag25/k
M=$W/runs_t1/lr1e-5/M/state; BG=$W/runs_e/B_g/state; R4DT=${R4DT:-float32}
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $K $W/status $W/logs; cd $W
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status_k.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
fp() { local got=$(sha256sum $1 | cut -c1-16); echo -e "fp\t$1\t$got\t$2" >> $W/status/status_k.tsv; [ "$got" = "$2" ]; }
CA="python $W/connect_audit.py --deterministic --train $P/gsm8k/train.jsonl --r3_dtype ${R3DT:-float64}"
if [ "$STAGE" = 1 ]; then
  fp $M/weights.pt 5d9825b68551462d || { echo "M 指纹不符"; exit 1; }
  go k_smoke bash -c "rm -rf /root/autodl-tmp/smoke_k && python $W/smoke_k.py --tok Qwen/Qwen2.5-0.5B-Instruct --work /root/autodl-tmp/smoke_k --fork_dir $W/fork --code_dir $W && rm -rf /root/autodl-tmp/smoke_k" || { echo "冒烟失败"; exit 1; }
  go k_legacy_run python $W/adam_offline_v3.py --state $M --act $W/diag25/b4/neuron_act_M.npz --out $K/legacy_v3_M.json --deterministic --perm_seeds 3
  go k_legacy_cmp python $W/compare_legacy.py --new $K/legacy_v3_M.json --ref $W/diag25/e1/adam_offline_v3_M.json --out $K/legacy_check.json
  go k_trial32 $CA --state $M --act $W/diag25/b4/neuron_act_M.npz --probe_new $P/gsm8k/probe.jsonl --probe_old $P/mbpp/probe.jsonl --out $K/trial_M_f32.json --nb 1 --n_eval 8 --r4_dtype float32
  go k_trial64 $CA --state $M --act $W/diag25/b4/neuron_act_M.npz --probe_new $P/gsm8k/probe.jsonl --probe_old $P/mbpp/probe.jsonl --out $K/trial_M_f64.json --nb 1 --n_eval 8 --r4_dtype float64
  touch $W/status/DONE_k1; echo "K stage 1 done $(date +%T)"
elif [ "$STAGE" = 2 ]; then
  go k_main_M $CA --state $M --act $W/diag25/b4/neuron_act_M.npz --probe_new $P/gsm8k/probe.jsonl --probe_old $P/mbpp/probe.jsonl --out $K/k_M.json --nb 8 --n_eval 32 --r4_dtype $R4DT
  touch $W/status/DONE_k2; echo "K stage 2 done $(date +%T)"
elif [ "$STAGE" = 3 ]; then
  fp $BG/weights.pt 70f6268bf66fd3f7 || { echo "B_g 指纹不符"; exit 1; }
  go k_main_PG $CA --state $BG --act $W/diag25/h/neuron_act_B_g_hall.npz --probe_new $P/gsm8k/probe.jsonl --probe_old $P/bio/probe.jsonl --out $K/k_PG.json --nb 8 --n_eval 32 --r4_dtype $R4DT
  touch $W/status/DONE_k3; echo "K stage 3 done $(date +%T)"
fi
go k_pack_$STAGE bash -c "cd $W; tar czf /root/autodl-tmp/k_results_$STAGE.tgz diag25/k logs/k_*.log status/status_k.tsv; ls -la /root/autodl-tmp/k_results_$STAGE.tgz"
