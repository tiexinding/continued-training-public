# 发布包：卡上运行队列的原始记录，仅供审计，不能直接运行（见本目录 README.md）；部分路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2），仍含卡上路径与清理命令
#!/bin/bash
# G4：带中间快照重训三臂（确定性重跑；终点 125 步与原运行对照），在 10/25/50/75 步做表示×读出拆分
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; E=$W/runs_e; P=$R/code/data/prepared_subsets; BG=$E/B_g/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/fork; mkdir -p $W/diag25/g2
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
rm -rf $E/E_anc_vt/snaps $E/E_anc_0/snaps
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl --probe bio=$P/bio/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
arm() { local n=$1; shift
  rm -rf $E/G4_$n
  go g4_train_$n python fork_continue.py fork --task gsm8k --train $P/gsm8k/train.jsonl --out $E/G4_$n --init $BG --epochs 1 --steps 125 --lr 5e-6 --order_seed 17 --full_eval 0,10,25,50,75,100,125 --snap 10,25,50,75 $C "$@"
  ST=""; for s in 0010 0025 0050 0075; do ST="$ST,s$s=$E/G4_$n/snaps/step$s"; done
  go g4_decomp_$n python $W/mech_diag.py --stage decomp --states "${ST#,}" --out $W/diag25/g2/$n --deterministic
  rm -rf $E/G4_$n/snaps; }
arm E_v  --m_op zero --v_op keep --t_reset
arm E_vt --m_op zero --v_op keep
arm E_0  --m_op zero --v_op reset --t_reset
go g4_fin bash -c "cd $W; tar czf /root/autodl-tmp/g4_results.tgz diag25/g2 runs_e/G4_*/log.jsonl logs/g4*.log; ls -la /root/autodl-tmp/g4_results.tgz"
touch $W/status/DONE_g4
