# 发布包：卡上运行队列的原始记录，仅供审计，不能直接运行（见本目录 README.md）；部分路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2），仍含卡上路径与清理命令
#!/bin/bash
# F9（修复显存：只算末位置 logits）+ F10（补 replay_data.py；func_cont 只留含代码的输出）；等 F11 结束后执行
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; E=$W/runs_e; P=$R/code/data/prepared_subsets; BG=$E/B_g/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/fork; mkdir -p $W/data_f $W/diag25/f2
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
until [ -f $W/status/DONE_f5 ]; do sleep 30; done
rm -f $W/status/DONE_f9_* $W/status/DONE_f10_* $W/status/DONE_f3 $W/status/DONE_f4
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl --probe bio=$P/bio/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
ST="E_vperm=$E/E_vperm/snaps/step0125,E_vt=$E/E_vt/snaps/step0125,E_vfac=$E/E_vfac/snaps/step0125,E_v=$E/E_v/snaps/step0125,E_0=$E/E_0/snaps/step0125"
go f9b_localize python $W/drift_diag.py --stage localize --deterministic --tok_mode manual --states "$ST"
go f10b_anchor_data python $W/anchor_data.py --state $BG --out $W/data_f/anchor_Bg.jsonl --deterministic
go f10b_mix python $W/replay_data.py --new $P/gsm8k/train.jsonl --old $W/data_f/anchor_Bg.jsonl --out $W/data_f/train_gsm8k_anchor10.jsonl --ratio 0.10
fork_a() { local n=$1; shift; go f10b_$n python fork_continue.py fork --task gsm8k --train $W/data_f/train_gsm8k_anchor10.jsonl --out $E/$n --init $BG --epochs 1 --steps 125 --lr 5e-6 --order_seed 17 --full_eval 0,25,50,75,100,125 --snap 125 $C "$@"; }
rm -rf $E/E_anc_vt $E/E_anc_0
fork_a E_anc_vt --m_op zero --v_op keep
fork_a E_anc_0  --m_op zero --v_op reset --t_reset
segnll() { local n=$1 st=$2 base=$3; printf '[{"name":"%s","base":"B","swap_from":null,"groups":[]}]\n' $n > $W/diag25/e4/cfg_$n.json
  go f10b_segnll_$n python state_mix_eval.py --stateA $base --stateB $st --data $P --configs $W/diag25/e4/cfg_$n.json --out $W/diag25/e4/segnll --nll; }
segnll E_anc_vt $E/E_anc_vt/snaps/step0125 $BG; segnll E_anc_0 $E/E_anc_0/snaps/step0125 $BG
go f10b_heval python $W/drift_diag.py --stage heval --deterministic --tok_mode manual --tag f10 --states "E_anc_vt=$E/E_anc_vt/snaps/step0125,E_anc_0=$E/E_anc_0/snaps/step0125"
go f10b_audit python $W/gen_audit.py --out $W/diag25/f2 --max_new 400 --deterministic --mbpp --states E_anc_vt=$E/E_anc_vt/snaps/step0125 E_anc_0=$E/E_anc_0/snaps/step0125
go f10b_fin bash -c "cd $W; tar czf /root/autodl-tmp/f10_results.tgz diag25/f1/localize.json diag25/f1/heval_f10.json diag25/f2 diag25/e4/segnll/E_anc_*.json runs_e/E_anc_*/log.jsonl data_f/*.jsonl logs/f9b*.log logs/f10b*.log status/status.tsv; ls -la /root/autodl-tmp/f10_results.tgz"
touch $W/status/DONE_f6
