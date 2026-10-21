# 发布包：卡上运行队列的原始记录，仅供审计，不能直接运行（见本目录 README.md）；部分路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2），仍含卡上路径与清理命令
#!/bin/bash
# D 轮（文档 31 第 4 节；10-07 准备）：D0 清盘 → D1 修正版离线反事实（M 起点 + 第二方向起点）→ D2 第二任务方向最小四臂（:matched）→ D3 成对分析 + 选择性重合 → D4 功能 → D5 干预 v3 → D6 收尾
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets; M=$W/runs_t1/lr1e-5/M/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/fork; mkdir -p $W/runs_d $W/status $W/logs $W/diag25/d1 $W/diag25/d3 $W/diag25/d4 $W/diag25/d5
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
REC="--dense 1,2,5,10,25,50,75,100 --fields --full_eval 0,10,25,50,75,100,108 --snap 108"
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
# ---- D0 清盘（删 C 轮非必需快照；保留 K_vfac/K_vperm/K_vt_s31/K_0_s31 终点）
go d0_clean bash -c "rm -rf $W/runs_c/K_vconst/snaps $W/runs_c/S_style/snaps $W/runs_c/S_official/snaps $W/runs_c/K_mvt_rec/snaps; df -h $W | tail -1; sha256sum $M/weights.pt | cut -c1-16"
# ---- D1 修正版离线反事实（M 起点，GSM8K 梯度）
go d1_offline_M python $W/adam_offline_v2.py --state $M --act $W/diag25/b4/neuron_act_M.npz --out $W/diag25/d1/adam_offline_v2_M.json --deterministic --perm_seeds 3
# ---- D2 第二任务方向：起点 A_g = W0 →GSM8K 5e-6 125 步（带优化器状态）；四臂 →MBPP 108 步；同任务对照 A_g →GSM8K 再 125 步
go d2_Ag python fork_continue.py segment --task gsm8k --train $P/gsm8k/train.jsonl --out $W/runs_d/A_g --init base --switch fresh --lr 5e-6 --epochs 1 --eval_every 25 $C
AG=$W/runs_d/A_g/state
go d2_act_Ag python $W/neuron_act.py --state $AG --name A_g --deterministic
go d1_offline_Ag python $W/adam_offline_v2.py --state $AG --act $W/diag25/b4/neuron_act_A_g.npz --out $W/diag25/d1/adam_offline_v2_Ag.json --deterministic --perm_seeds 3 --train $P/mbpp/train.jsonl
fork_m() { local n=$1; shift; go d2_$n python fork_continue.py fork --task mbpp --train $P/mbpp/train.jsonl --out $W/runs_d/$n --init $AG --epochs 1 --steps 108 --lr 5e-6 --order_seed 17 $REC $C "$@"; }
fork_m R_vt    --m_op zero --v_op keep
fork_m R_vfac  --m_op zero --v_op ladder:rowcol:matched
fork_m R_vperm --m_op zero --v_op ladder:cperm:matched
fork_m R_0     --m_op zero --v_op reset --t_reset
go d2_Rc python fork_continue.py segment --task gsm8k --train $P/gsm8k/train.jsonl --out $W/runs_d/R_c --init $AG --switch keep --lr 5e-6 --epochs 1 --eval_every 25 --order_seed 23 $C
# ---- D3 成对分析（A=A_g，B=臂，C=R_c 同任务）+ 选择性重合（A_g 的激活图；注意此处"新任务"= MBPP，选择性符号相反，在本地分析时翻转）
for A in R_vt R_vfac R_vperm R_0; do
  T=$W/runs_d/pairs_$A; mkdir -p $T/A $T/B $T/C; ln -sfn $AG $T/A/state; ln -sfn $W/runs_d/$A/snaps/step0108 $T/B/state; ln -sfn $W/runs_d/R_c/state $T/C/state
  go d3_pair_$A /root/miniconda3/bin/python $W/phase2_card.py --phase3 $T --out $W/diag25/d3/pairs_$A --device cuda
done
go d3_neuron python $W/neuron_act.py --state $AG --name A_g_d --deterministic --fields $(for A in R_vt R_vfac R_vperm R_0; do echo -n "$A=$W/diag25/d3/pairs_$A/A_B_fields.npz "; done)
# ---- D4 功能
STATES=""; for A in R_vt R_vfac R_vperm R_0; do STATES="$STATES $A=$W/runs_d/$A/snaps/step0108"; done; STATES="$STATES A_g=$AG R_c=$W/runs_d/R_c/state"
go d4_audit python $W/gen_audit.py --out $W/diag25/d4 --max_new 400 --deterministic --mbpp --states $STATES
go d4_humaneval python $W/humaneval_eval.py --out $W/diag25/d4/humaneval_d.json --states $STATES --deterministic
go d4_dolly python $W/dolly_eval.py --out $W/diag25/d4/dolly_gen_d.json --states $STATES --deterministic
segnll() { local n=$1 st=$2; printf '[{"name":"%s","base":"B","swap_from":null,"groups":[]}]\n' $n > $W/diag25/d4/cfg_$n.json
  go d4_segnll_$n python state_mix_eval.py --stateA $AG --stateB $st --data $P --configs $W/diag25/d4/cfg_$n.json --out $W/diag25/d4/segnll --nll; }
for A in R_vt R_vfac R_vperm R_0; do segnll $A $W/runs_d/$A/snaps/step0108; done; segnll R_c $W/runs_d/R_c/state; segnll A_g $AG
# ---- D5 干预 v3（M →GSM8K 的 K_mvt 终态；纯 NLL；独立代码集 HumanEval 规范解）
go d5_intervene python $W/intervene_v3.py --A $M --B $W/runs_t2/lr5e-6/G/state --tag Gkeep_v3 --fields $W/diag25/t2/lr5e-6/pairs/A_B_fields.npz --act $W/diag25/b4/neuron_act_M.npz --v3 --n_random 5 --fracs 0.05 --out $W/diag25/d5 --no_gen --deterministic
# ---- D6 收尾
go d6_fin bash -c "rm -f $W/runs_d/R_c/state/optim.pt; for A in R_vt R_vfac R_vperm R_0; do sha256sum $W/runs_d/\$A/snaps/step0108/weights.pt; done; sha256sum $AG/weights.pt $W/runs_d/R_c/state/weights.pt; cd $W; tar czf /root/autodl-tmp/d_results_small.tgz diag25/d1 diag25/d4 diag25/d5 diag25/b4/neuron_act_A_g*.json diag25/b4/neuron_act_A_g_d.json \$(ls diag25/d3/pairs_*/*.json) runs_d/*/log.jsonl logs/d*.log status/status.tsv; ls -la /root/autodl-tmp/d_results_small.tgz"
echo "D all done $(date +%T)"; touch $W/status/DONE_d
