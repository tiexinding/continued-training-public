# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# 统一开卡队列 C0–C7（文档 28）。串行；每项有 DONE 标记与 rc；失败跳过。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets; M=$W/runs_t1/lr1e-5/M/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/fork; mkdir -p $W/runs_c $W/status $W/logs $W/diag25/c1 $W/diag25/c3 $W/diag25/c4 $W/diag25/c5 $W/diag25/c6 $W/data_c
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
REC="--dense 1,2,5,10,25,50,75,100,125 --fields --full_eval 0,10,25,50,75,100,125 --snap 125"
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
ACT=$W/diag25/b4/neuron_act_M.npz
# ---- C0 清盘与校验
go c0_clean bash -c "rm -f $W/runs/A/state/optim.pt $W/runs/B/state/optim.pt; rm -rf $W/runs_b2/Kv/snaps/step0010 $W/runs_b2/Kv/snaps/step0050 $W/runs_b2/bridge/snaps/step0010 $W/runs_b2/bridge/snaps/step0050 $W/runs_b2/Cfresh/snaps; df -h $W | tail -1; sha256sum $M/weights.pt $W/runs_t2/lr5e-6/G/state/weights.pt $W/runs_t3/lr5e-6/G/state/weights.pt $W/runs_t2/lr5e-6/C/state/weights.pt"
# ---- C1 离线机制（Q1/Q2）
go c1_adam_offline python $W/adam_offline.py --state $M --act $ACT --out $W/diag25/c1/adam_offline.json --deterministic
# ---- C2 前置：风格数据（Q7）
go c2_style_data python $W/style_data.py --out_dir $W/data_c --deterministic
# ---- C2 训练臂（均从 M；→GSM8K 125 步 / →MBPP 108 步；lr 5e-6）
fork_g() { local n=$1; shift; go c2_$n python fork_continue.py fork --task gsm8k --train $P/gsm8k/train.jsonl --out $W/runs_c/$n --init $M --epochs 1 --steps 125 --lr 5e-6 $REC $C "$@"; }
fork_g K_mvt_rec --m_op scale:1 --v_op keep --order_seed 17
fork_g K_vt_s31  --m_op zero --v_op keep --order_seed 31
fork_g K_0_s31   --m_op zero --v_op reset --t_reset --order_seed 31
fork_g K_vperm   --m_op zero --v_op ladder:cperm --order_seed 17
fork_g K_vfac    --m_op zero --v_op ladder:rowcol --order_seed 17
fork_g K_vconst  --m_op zero --v_op ladder:const --order_seed 17
go c2_S_style python fork_continue.py fork --task gsm8k --train $W/data_c/train_style.jsonl --out $W/runs_c/S_style --init $M --epochs 2 --steps 125 --lr 5e-6 --m_op scale:1 --v_op keep --order_seed 17 --full_eval 0,10,25,50,75,100,125 --snap 125 $C
go c2_S_official python fork_continue.py fork --task gsm8k --train $W/data_c/train_official_same.jsonl --out $W/runs_c/S_official --init $M --epochs 2 --steps 125 --lr 5e-6 --m_op scale:1 --v_op keep --order_seed 17 --full_eval 0,10,25,50,75,100,125 --snap 125 $C
# ---- C3 成对分析 + 选择性重合
for A in K_mvt_rec K_vt_s31 K_0_s31 K_vperm K_vfac K_vconst S_style S_official; do
  T=$W/runs_c/pairs_$A; mkdir -p $T/A $T/B $T/C; ln -sfn $M $T/A/state; ln -sfn $W/runs_c/$A/snaps/step0125 $T/B/state; ln -sfn $W/runs_t2/lr5e-6/C/state $T/C/state
  go c3_pair_$A /root/miniconda3/bin/python $W/phase2_card.py --phase3 $T --out $W/diag25/c3/pairs_$A --device cuda
done
go c3_neuron python $W/neuron_act.py --state $M --name M_c --deterministic --fields $(for A in K_mvt_rec K_vt_s31 K_0_s31 K_vperm K_vfac K_vconst S_style S_official; do echo -n "$A=$W/diag25/c3/pairs_$A/A_B_fields.npz "; done)
# ---- C4 功能评估
STATES=""; for A in K_mvt_rec K_vt_s31 K_0_s31 K_vperm K_vfac K_vconst S_style S_official; do STATES="$STATES $A=$W/runs_c/$A/snaps/step0125"; done
OLD="K_vt=$W/runs_b2/Kv/snaps/step0125 K_t=$W/runs_b2/bridge/snaps/step0125 C_0=$W/runs_b2/Cfresh/state C31=$W/runs_b3/C31/state"
go c4_audit python $W/gen_audit.py --out $W/diag25/c4 --max_new 400 --deterministic --mbpp --states $STATES
go c4_humaneval python $W/humaneval_eval.py --out $W/diag25/c4/humaneval_c.json --states $STATES $OLD --deterministic
go c4_dolly python $W/dolly_eval.py --out $W/diag25/c4/dolly_gen.json --states W0=base M=$M K_mvt=$W/runs_t2/lr5e-6/G/state K_0=$W/runs_t3/lr5e-6/G/state C23=$W/runs_t2/lr5e-6/C/state $OLD $STATES --deterministic
segnll() { local n=$1 st=$2; printf '[{"name":"%s","base":"B","swap_from":null,"groups":[]}]\n' $n > $W/diag25/c4/cfg_$n.json
  go c4_segnll_$n python state_mix_eval.py --stateA $M --stateB $st --data $P --configs $W/diag25/c4/cfg_$n.json --out $W/diag25/c4/segnll --nll; }
for A in K_mvt_rec K_vt_s31 K_0_s31 K_vperm K_vfac K_vconst S_style S_official; do segnll $A $W/runs_c/$A/snaps/step0125; done
segnll K_vt $W/runs_b2/Kv/snaps/step0125; segnll K_t $W/runs_b2/bridge/snaps/step0125; segnll C_0 $W/runs_b2/Cfresh/state; segnll C31 $W/runs_b3/C31/state
# ---- C5 干预 v2（Q5）
go c5_intervene python $W/intervene_v2.py --A $M --B $W/runs_t2/lr5e-6/G/state --tag Gkeep_v2 --fields $W/diag25/t2/lr5e-6/pairs/A_B_fields.npz --act $ACT --v2 --n_random 5 --fracs 0.05 --out $W/diag25/c5 --deterministic
# ---- C6 梯度 mask 消融（Q6）
for MK in answer answer,calc calc; do go c6_mask_$(echo $MK | tr , _) python $W/grad_geom_mask.py --state $M --out $W/diag25/c6/grad_mask_$(echo $MK | tr , _).json --mask $MK --deterministic; done
# ---- C7 收尾
go c7_fin bash -c "for A in K_mvt_rec K_vt_s31 K_0_s31 K_vperm K_vfac K_vconst S_style S_official; do sha256sum $W/runs_c/\$A/snaps/step0125/weights.pt; done; cd $W; tar czf /root/autodl-tmp/c_results.tgz diag25/c1 diag25/c3 diag25/c4 diag25/c5 diag25/c6 runs_c/*/log.jsonl runs_c/K_mvt_rec/records runs_c/K_vt_s31/records runs_c/K_0_s31/records runs_c/K_vperm/records data_c/meta.json logs status; ls -la /root/autodl-tmp/c_results.tgz"
echo "C all done $(date +%T)"; touch $W/status/DONE_c
