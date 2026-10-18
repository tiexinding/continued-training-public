# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# F10 自锚定两臂复跑：混合集 968 行只够 121 步，快照改在第 121 步；随后评估四臂（含 F12 两臂的 heval）
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; E=$W/runs_e; P=$R/code/data/prepared_subsets; BG=$E/B_g/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/fork
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl --probe bio=$P/bio/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
rm -rf $E/E_anc_vt $E/E_anc_0
fork_x() { local n=$1; shift; go f14_$n python fork_continue.py fork --task gsm8k --train $W/data_f/train_gsm8k_anchor10.jsonl --out $E/$n --init $BG --epochs 1 --steps 121 --lr 5e-6 --order_seed 17 --full_eval 0,25,50,75,100,121 --snap 121 $C "$@"; }
fork_x E_anc_vt --m_op zero --v_op keep
fork_x E_anc_0  --m_op zero --v_op reset --t_reset
segnll() { local n=$1 st=$2 base=$3; printf '[{"name":"%s","base":"B","swap_from":null,"groups":[]}]\n' $n > $W/diag25/e4/cfg_$n.json
  go f14_segnll_$n python state_mix_eval.py --stateA $base --stateB $st --data $P --configs $W/diag25/e4/cfg_$n.json --out $W/diag25/e4/segnll --nll; }
segnll E_anc_vt $E/E_anc_vt/snaps/step0121 $BG; segnll E_anc_0 $E/E_anc_0/snaps/step0121 $BG
go f14_heval python $W/drift_diag.py --stage heval --deterministic --tok_mode manual --tag f14 --states "E_vfac_t0=$E/E_vfac_t0/snaps/step0125,E_vperm_t0=$E/E_vperm_t0/snaps/step0125,E_anc_vt=$E/E_anc_vt/snaps/step0121,E_anc_0=$E/E_anc_0/snaps/step0121"
go f14_audit python $W/gen_audit.py --out $W/diag25/f2 --max_new 400 --deterministic --mbpp --states E_anc_vt=$E/E_anc_vt/snaps/step0121 E_anc_0=$E/E_anc_0/snaps/step0121 E_vfac_t0=$E/E_vfac_t0/snaps/step0125 E_vperm_t0=$E/E_vperm_t0/snaps/step0125
go f14_fin bash -c "cd $W; tar czf /root/autodl-tmp/f14_results.tgz diag25/f1/*.json diag25/f2 diag25/e4/segnll/E_anc_*.json diag25/e4/segnll/E_v*_t0.json diag25/e3/pairs_E_v*_t0/*.json diag25/b4/neuron_act_B_g_f1*.json runs_e/E_anc_*/log.jsonl runs_e/E_v*_t0/log.jsonl runs_e/E_vclamp*/log.jsonl runs_e/E_v/log.jsonl data_f/*.jsonl logs/f9b*.log logs/f1[0-4]*.log status/status.tsv; ls -la /root/autodl-tmp/f14_results.tgz"
touch $W/status/DONE_f9x
