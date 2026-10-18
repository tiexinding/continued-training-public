# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# 批次 4（10-06）：神经元级分析与局部干预（在批次 3 之后串行，避免 GPU/CPU 争用）。
#  W4-1/2 neuron_act：M 上三任务探针的前馈中间神经元激活统计 + 与 M→Gkeep / M→Gfresh / M→Kv / M→bridge 改动排序的重合；
#  W4-3 intervene：从 Gkeep（及 Gfresh）撤回 top-1%/5% 神经元改动，带随机 / 能量匹配随机 / 插值对照，评估 NLL + 生成。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; M=$W/runs_t1/lr1e-5/M/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4
cd $W/fork; mkdir -p $W/status $W/logs $W/diag25/b4
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
FK=$W/diag25/t2/lr5e-6/pairs/A_B_fields.npz; FF=$W/diag25/t3/pairs/A_B_fields.npz
EXTRA=""; [ -f $W/diag25/b2/pairs_M/A_B_fields.npz ] && EXTRA="Kv=$W/diag25/b2/pairs_M/A_B_fields.npz"
go b4_act python $W/neuron_act.py --state $M --name M --deterministic --fields Gkeep=$FK Gfresh=$FF $EXTRA
go b4_int_keep python $W/intervene.py --A $M --B $W/runs_t2/lr5e-6/G/state --tag Gkeep --fields $FK --act $W/diag25/b4/neuron_act_M.npz --deterministic
go b4_int_fresh python $W/intervene.py --A $M --B $W/runs_t3/lr5e-6/G/state --tag Gfresh --fields $FF --act $W/diag25/b4/neuron_act_M.npz --deterministic
echo "B4 all done $(date +%T)"; touch $W/status/DONE_b4
