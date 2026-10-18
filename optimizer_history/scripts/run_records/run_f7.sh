# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# F12（老丁问"vfac 能否去掉 t"）：行列近似 + 清 t（E_vfac_t0）、通道置换 + 清 t（E_vperm_t0）；评估同 F7 / F11。匹配系数在清 t 之前按旧 t 计算。
# 评估：成对结构、选择性重合、分段 NLL、首位置概率 + HumanEval 两规则（含旧四臂同口径复测）。最后回传记录。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; E=$W/runs_e; P=$R/code/data/prepared_subsets; BG=$E/B_g/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/fork; mkdir -p $W/diag25/f2 $W/diag25/f1
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
until [ -f $W/status/DONE_f6 ]; do sleep 30; done
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl --probe bio=$P/bio/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
REC="--dense 1,2,5,10,25,50,75,100,125 --fields --full_eval 0,10,25,50,75,100,125 --snap 125"
fork_g() { local n=$1; shift; go f12_$n python fork_continue.py fork --task gsm8k --train $P/gsm8k/train.jsonl --out $E/$n --init $BG --epochs 1 --steps 125 --lr 5e-6 --order_seed 17 $REC $C "$@"; }
fork_g E_vfac_t0 --m_op zero --v_op ladder:rowcol:matched_med --t_reset
fork_g E_vperm_t0 --m_op zero --v_op ladder:cperm:matched_med --t_reset
for A in E_vfac_t0 E_vperm_t0; do
  T=$E/pairs_$A; mkdir -p $T/A $T/B $T/C; ln -sfn $BG $T/A/state; ln -sfn $E/$A/snaps/step0125 $T/B/state; ln -sfn $E/B_c/state $T/C/state
  go f12_pair_$A /root/miniconda3/bin/python $W/phase2_card.py --phase3 $T --out $W/diag25/e3/pairs_$A --device cuda
done
go f12_neuron python $W/neuron_act.py --state $BG --name B_g_f12 --deterministic --tasks gsm8k,mbpp,dolly_qa,bio --sel gsm8k,bio --fields $(for A in E_vfac_t0 E_vperm_t0 E_vt E_0; do echo -n "$A=$W/diag25/e3/pairs_$A/A_B_fields.npz "; done)
segnll() { local n=$1 st=$2 base=$3; printf '[{"name":"%s","base":"B","swap_from":null,"groups":[]}]\n' $n > $W/diag25/e4/cfg_$n.json
  go f12_segnll_$n python state_mix_eval.py --stateA $base --stateB $st --data $P --configs $W/diag25/e4/cfg_$n.json --out $W/diag25/e4/segnll --nll; }
segnll E_vfac_t0 $E/E_vfac_t0/snaps/step0125 $BG; segnll E_vperm_t0 $E/E_vperm_t0/snaps/step0125 $BG
go f12_heval python $W/drift_diag.py --stage heval --deterministic --tok_mode manual --tag f12 --states "B_g=$BG,E_vfac_t0=$E/E_vfac_t0/snaps/step0125,E_v=$E/E_v/snaps/step0125,E_vperm_t0=$E/E_vperm_t0/snaps/step0125,E_vt=$E/E_vt/snaps/step0125,E_vfac=$E/E_vfac/snaps/step0125,E_vperm=$E/E_vperm/snaps/step0125,E_0=$E/E_0/snaps/step0125"
go f12_energy python $W/drift_diag.py --stage energy --deterministic --tok_mode manual --pairs "E_vfac_t0=$BG:$E/E_vfac_t0/snaps/step0125,E_vperm_t0=$BG:$E/E_vperm_t0/snaps/step0125"
cp $W/diag25/f1/energy_enrich.json $W/diag25/f2/energy_enrich_f12.json
go f12_fin bash -c "cd $W; tar czf /root/autodl-tmp/f12_results.tgz diag25/f1 diag25/f2 diag25/e3/pairs_E_vfac_t0/*.json diag25/e3/pairs_E_vperm_t0/*.json diag25/b4/neuron_act_B_g_f.json diag25/e4/segnll/E_vfac_t0.json diag25/e4/segnll/E_vperm_t0.json runs_e/E_vfac_t0/log.jsonl runs_e/E_vperm_t0/log.jsonl runs_e/E_*/records/scalars_*.pt runs_e/E_*/records/fullu_*.pt logs/f*.log status/status.tsv; ls -la /root/autodl-tmp/f_results.tgz"
echo "F2 all done $(date +%T)"; touch $W/status/DONE_f7x
