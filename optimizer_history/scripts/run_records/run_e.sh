# 发布包：卡上运行队列的原始记录，仅供审计，不能直接运行（见本目录 README.md）；部分路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2），仍含卡上路径与清理命令
#!/bin/bash
# E 轮（文档 31 第 9 节 P1–P4；文档 34）：E0 清盘上传校验 → E1 离线 v3（M、A_g）→ E2 真正新任务对 Bio→GSM8K 四臂（matched_med）+ 同任务对照 + 第二方向 R_vperm 用 matched_med 复跑 → E3 回放两臂 → E4 评估 → E5 独立同域确认干预 → E6 收尾
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets; M=$W/runs_t1/lr1e-5/M/state; AG=$W/runs_d/A_g/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/fork; mkdir -p $W/runs_e $W/status $W/logs $W/diag25/e1 $W/diag25/e3 $W/diag25/e4 $W/diag25/e5 $W/data_e
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl --probe bio=$P/bio/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
REC="--dense 1,2,5,10,25,50,75,100,125 --fields --full_eval 0,10,25,50,75,100,125 --snap 125 --full_u_layers 6,18"
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
# ---- E0
go e0_clean bash -c "rm -rf $W/runs_c/K_vperm/snaps $W/runs_c/K_vfac/snaps $W/runs_b2/Cfresh/state $W/runs_d/R_vt/records/fields_*.pt $W/runs_d/R_vfac/records/fields_*.pt $W/runs_d/R_vperm/records/fields_*.pt $W/runs_d/R_0/records; df -h $W | tail -1; sha256sum $M/weights.pt $AG/weights.pt | cut -c1-16"
# ---- E1 离线 v3（P1）：仅行/仅列/分布匹配随机因子；两起点
go e1_offline_M  python $W/adam_offline_v3.py --state $M  --act $W/diag25/b4/neuron_act_M.npz   --out $W/diag25/e1/adam_offline_v3_M.json  --deterministic --perm_seeds 3
go e1_offline_Ag python $W/adam_offline_v3.py --state $AG --act $W/diag25/b4/neuron_act_A_g.npz --out $W/diag25/e1/adam_offline_v3_Ag.json --deterministic --perm_seeds 3 --train $P/mbpp/train.jsonl
# ---- E2 真正新任务对：B_g = W0 →Bio（PubMed 题→摘要）5e-6 一轮 125 步；四臂 →GSM8K 125 步；同任务对照 B_c = B_g →Bio 再 125 步
go e2_Bg python fork_continue.py segment --task bio --train $P/bio/train.jsonl --out $W/runs_e/B_g --init base --switch fresh --lr 5e-6 --epochs 1 --eval_every 25 $C
BG=$W/runs_e/B_g/state
go e2_act_Bg python $W/neuron_act.py --state $BG --name B_g --deterministic
fork_g() { local n=$1; shift; go e2_$n python fork_continue.py fork --task gsm8k --train $P/gsm8k/train.jsonl --out $W/runs_e/$n --init $BG --epochs 1 --steps 125 --lr 5e-6 --order_seed 17 $REC $C "$@"; }
fork_g E_vt    --m_op zero --v_op keep
fork_g E_vfac  --m_op zero --v_op ladder:rowcol:matched_med
fork_g E_vperm --m_op zero --v_op ladder:cperm:matched_med
fork_g E_0     --m_op zero --v_op reset --t_reset
go e2_Bc python fork_continue.py segment --task bio --train $P/bio/train.jsonl --out $W/runs_e/B_c --init $BG --switch keep --lr 5e-6 --epochs 1 --eval_every 25 --order_seed 23 $C
rm -f $W/runs_e/B_c/state/optim.pt
# 第二方向 R_vperm 用稳健匹配复跑（修正 D 轮步幅偏小）
go e2_R_vperm_med python fork_continue.py fork --task mbpp --train $P/mbpp/train.jsonl --out $W/runs_e/R_vperm_med --init $AG --epochs 1 --steps 108 --lr 5e-6 --order_seed 17 --dense 1,2,5,10,25,50,75,100 --fields --full_eval 0,10,25,50,75,100,108 --snap 108 --m_op zero --v_op ladder:cperm:matched_med $C
# ---- E3 回放两臂（P3）：从 M →GSM8K，训练集 = 90% GSM8K + 10% MBPP token（固定总 token），125 步
go e3_P_vt python fork_continue.py fork --task gsm8k --train $W/data_e/train_gsm8k_replay10.jsonl --out $W/runs_e/P_vt --init $M --epochs 1 --steps 125 --lr 5e-6 --order_seed 17 --full_eval 0,10,25,50,75,100,125 --snap 125 --m_op zero --v_op keep $C
go e3_P_0  python fork_continue.py fork --task gsm8k --train $W/data_e/train_gsm8k_replay10.jsonl --out $W/runs_e/P_0  --init $M --epochs 1 --steps 125 --lr 5e-6 --order_seed 17 --full_eval 0,10,25,50,75,100,125 --snap 125 --m_op zero --v_op reset --t_reset $C
# ---- E4 评估：成对分析（A=B_g，B=臂，C=B_c）、选择性重合、生成、HumanEval、Dolly、逐样本 NLL（含 bio 探针）
for A in E_vt E_vfac E_vperm E_0; do
  T=$W/runs_e/pairs_$A; mkdir -p $T/A $T/B $T/C; ln -sfn $BG $T/A/state; ln -sfn $W/runs_e/$A/snaps/step0125 $T/B/state; ln -sfn $W/runs_e/B_c/state $T/C/state
  go e4_pair_$A /root/miniconda3/bin/python $W/phase2_card.py --phase3 $T --out $W/diag25/e3/pairs_$A --device cuda
done
go e4_neuron python $W/neuron_act.py --state $BG --name B_g_e --deterministic --fields $(for A in E_vt E_vfac E_vperm E_0; do echo -n "$A=$W/diag25/e3/pairs_$A/A_B_fields.npz "; done)
T=$W/runs_e/pairs_R_vperm_med; mkdir -p $T/A $T/B $T/C; ln -sfn $AG $T/A/state; ln -sfn $W/runs_e/R_vperm_med/snaps/step0108 $T/B/state; ln -sfn $W/runs_d/R_c/state $T/C/state
go e4_pair_R_vperm_med /root/miniconda3/bin/python $W/phase2_card.py --phase3 $T --out $W/diag25/e3/pairs_R_vperm_med --device cuda
go e4_neuron_R python $W/neuron_act.py --state $AG --name A_g_e --deterministic --fields R_vperm_med=$W/diag25/e3/pairs_R_vperm_med/A_B_fields.npz
STATES="B_g=$BG E_vt=$W/runs_e/E_vt/snaps/step0125 E_vfac=$W/runs_e/E_vfac/snaps/step0125 E_vperm=$W/runs_e/E_vperm/snaps/step0125 E_0=$W/runs_e/E_0/snaps/step0125 B_c=$W/runs_e/B_c/state P_vt=$W/runs_e/P_vt/snaps/step0125 P_0=$W/runs_e/P_0/snaps/step0125 R_vperm_med=$W/runs_e/R_vperm_med/snaps/step0108"
go e4_audit python $W/gen_audit.py --out $W/diag25/e4 --max_new 400 --deterministic --mbpp --states $STATES
go e4_humaneval python $W/humaneval_eval.py --out $W/diag25/e4/humaneval_e.json --states $STATES --deterministic
go e4_dolly python $W/dolly_eval.py --out $W/diag25/e4/dolly_gen_e.json --states $STATES --deterministic
segnll() { local n=$1 st=$2 base=$3; printf '[{"name":"%s","base":"B","swap_from":null,"groups":[]}]\n' $n > $W/diag25/e4/cfg_$n.json
  go e4_segnll_$n python state_mix_eval.py --stateA $base --stateB $st --data $P --configs $W/diag25/e4/cfg_$n.json --out $W/diag25/e4/segnll --nll; }
for A in E_vt E_vfac E_vperm E_0; do segnll $A $W/runs_e/$A/snaps/step0125 $BG; done; segnll B_c $W/runs_e/B_c/state $BG; segnll B_g $BG $BG
segnll P_vt $W/runs_e/P_vt/snaps/step0125 $M; segnll P_0 $W/runs_e/P_0/snaps/step0125 $M; segnll R_vperm_med $W/runs_e/R_vperm_med/snaps/step0108 $AG
# ---- E5 独立同域确认（P4）：选择性只用探针前 50 题估计，确认集 = 后 50 题
go e5_act_est python $W/neuron_act.py --state $M --name M_est50 --deterministic --probe_slice 0:50
go e5_intervene python $W/intervene_v3.py --A $M --B $W/runs_t2/lr5e-6/G/state --tag Gkeep_v3_confirm --fields $W/diag25/t2/lr5e-6/pairs/A_B_fields.npz --act $W/diag25/b4/neuron_act_M_est50.npz --v3 --n_random 5 --fracs 0.05 --out $W/diag25/e5 --no_gen --eval_slice 50:100 --deterministic
# ---- E6 收尾
go e6_fin bash -c "rm -f $W/runs_e/B_g/state/optim.pt.keep; for A in E_vt E_vfac E_vperm E_0 P_vt P_0; do sha256sum $W/runs_e/\$A/snaps/step0125/weights.pt; done; sha256sum $BG/weights.pt $W/runs_e/B_c/state/weights.pt $W/runs_e/R_vperm_med/snaps/step0108/weights.pt; cd $W; tar czf /root/autodl-tmp/e_results_small.tgz diag25/e1 diag25/e4 diag25/e5 diag25/b4/neuron_act_B_g*.json diag25/b4/neuron_act_A_g_e.json diag25/b4/neuron_act_M_est50.json \$(ls diag25/e3/pairs_*/*.json) runs_e/*/log.jsonl runs_e/*/records/match_c.pt logs/e*.log status/status.tsv; ls -la /root/autodl-tmp/e_results_small.tgz"
echo "E all done $(date +%T)"; touch $W/status/DONE_e
