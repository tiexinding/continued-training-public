# 发布包：卡上运行队列的原始记录，仅供审计，不能直接运行（见本目录 README.md）；部分路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2），仍含卡上路径与清理命令
#!/bin/bash
# F10b/F12 复跑（磁盘满后）：只存终态快照，不存逐步场记录
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
fork_x() { local n=$1 tr=$2; shift 2; go f13_$n python fork_continue.py fork --task gsm8k --train $tr --out $E/$n --init $BG --epochs 1 --steps 125 --lr 5e-6 --order_seed 17 --full_eval 0,25,50,75,100,125 --snap 125 $C "$@"; }
fork_x E_vfac_t0  $P/gsm8k/train.jsonl --m_op zero --v_op ladder:rowcol:matched_med --t_reset
fork_x E_vperm_t0 $P/gsm8k/train.jsonl --m_op zero --v_op ladder:cperm:matched_med --t_reset
fork_x E_anc_vt $W/data_f/train_gsm8k_anchor10.jsonl --m_op zero --v_op keep
fork_x E_anc_0  $W/data_f/train_gsm8k_anchor10.jsonl --m_op zero --v_op reset --t_reset
segnll() { local n=$1 st=$2 base=$3; printf '[{"name":"%s","base":"B","swap_from":null,"groups":[]}]\n' $n > $W/diag25/e4/cfg_$n.json
  go f13_segnll_$n python state_mix_eval.py --stateA $base --stateB $st --data $P --configs $W/diag25/e4/cfg_$n.json --out $W/diag25/e4/segnll --nll; }
for A in E_vfac_t0 E_vperm_t0 E_anc_vt E_anc_0; do segnll $A $E/$A/snaps/step0125 $BG; done
go f13_heval python $W/drift_diag.py --stage heval --deterministic --tok_mode manual --tag f13 --states "E_vfac_t0=$E/E_vfac_t0/snaps/step0125,E_vperm_t0=$E/E_vperm_t0/snaps/step0125,E_anc_vt=$E/E_anc_vt/snaps/step0125,E_anc_0=$E/E_anc_0/snaps/step0125"
for A in E_vfac_t0 E_vperm_t0; do
  T=$E/pairs_$A; mkdir -p $T/A $T/B $T/C; ln -sfn $BG $T/A/state; ln -sfn $E/$A/snaps/step0125 $T/B/state; ln -sfn $E/B_c/state $T/C/state
  go f13_pair_$A /root/miniconda3/bin/python $W/phase2_card.py --phase3 $T --out $W/diag25/e3/pairs_$A --device cuda
done
go f13_neuron python $W/neuron_act.py --state $BG --name B_g_f13 --deterministic --tasks gsm8k,mbpp,dolly_qa,bio --sel gsm8k,bio --fields $(for A in E_vfac_t0 E_vperm_t0; do echo -n "$A=$W/diag25/e3/pairs_$A/A_B_fields.npz "; done)
go f13_audit python $W/gen_audit.py --out $W/diag25/f2 --max_new 400 --deterministic --mbpp --states E_anc_vt=$E/E_anc_vt/snaps/step0125 E_anc_0=$E/E_anc_0/snaps/step0125
go f13_fin bash -c "cd $W; tar czf /root/autodl-tmp/f13_results.tgz diag25/f1/*.json diag25/f2 diag25/e4/segnll/E_anc_*.json diag25/e4/segnll/E_v*_t0.json diag25/e3/pairs_E_v*_t0/*.json diag25/b4/neuron_act_B_g_f1*.json runs_e/E_anc_*/log.jsonl runs_e/E_v*_t0/log.jsonl runs_e/E_vclamp*/log.jsonl runs_e/E_v/log.jsonl data_f/*.jsonl logs/f9b*.log logs/f1[0-3]*.log status/status.tsv; ls -la /root/autodl-tmp/f13_results.tgz"
touch $W/status/DONE_f8
