# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# H 轮（文档 41 v3，10-09 开卡）：H0 指纹 + 冒烟 + 确定性复跑 → H5′ 种子 31（10 条）→ H3 种子 17 中位数校准补齐（3 条）→ H5″ 种子 47（12 条）
# 每条运行流水：训练（--upd_energy，H2′）→ 成对分析 A→B（--ab_only，含平衡分解收敛记录）→ HumanEval（两规则 + 无缩进率）→ 逐样本损失 → 删终点快照
# 磁盘：卡上仅余约 12 G，快照评估后即删，峰值约一份权重（2 G）。串行，FORK_THREADS=4。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets
M=$W/runs_t1/lr1e-5/M/state; AG=$W/runs_d/A_g/state; BG=$W/runs_e/B_g/state
H=$W/runs_h; D=$W/diag25/h
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/fork; mkdir -p $H $D/segnll $W/status $W/logs
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
abort() { echo "ABORT: $*"; echo -e "ABORT\t$*\t$(date +%T)" >> $W/status/status.tsv; touch $W/status/H_ABORT; exit 1; }
# 探针集与原运行一致：MG、GM 三个（C/D 轮），PG 四个（E 轮）
PR3="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl"
PR4="$PR3 --probe bio=$P/bio/probe.jsonl"

# 切换：起点 / 新任务 / 步数 / 探针
sw_init()  { case $1 in MG) echo $M;; GM) echo $AG;; PG) echo $BG;; esac; }
sw_task()  { case $1 in MG|PG) echo gsm8k;; GM) echo mbpp;; esac; }
sw_steps() { case $1 in MG|PG) echo 125;; GM) echo 108;; esac; }
sw_pr()    { case $1 in PG) echo "$PR4";; *) echo "$PR3";; esac; }
# 条件：四个都清零 m；两种近似都用中位数校准
cond_op()  { case $1 in vt) echo "--m_op zero --v_op keep";; vfac) echo "--m_op zero --v_op ladder:rowcol:matched_med";;
                        vperm) echo "--m_op zero --v_op ladder:cperm:matched_med";; 0) echo "--m_op zero --v_op reset --t_reset";; esac; }

train() {   # train <sw> <cond> <seed> <outname>
  local sw=$1 cd=$2 sd=$3 n=$4; local st=$(sw_steps $sw)
  [ -f $W/status/DONE_h_train_$n ] || rm -rf $H/$n
  go h_train_$n python fork_continue.py fork --task $(sw_task $sw) --train $P/$(sw_task $sw)/train.jsonl --out $H/$n --init $(sw_init $sw) \
     --epochs 1 --steps $st --lr 5e-6 --order_seed $sd --full_eval 0,25,50,75,100,$st --dense "" --snap $st --upd_energy \
     --tok_mode manual --deterministic $(sw_pr $sw) $(cond_op $cd); }

readout() { # readout <sw> <n>：成对分析、HumanEval、逐样本损失，之后删快照
  local sw=$1 n=$2; local S=$H/$n/snaps/step$(printf %04d $(sw_steps $sw)); local A=$(sw_init $sw)
  [ -f $S/weights.pt ] || { [ -f $W/status/DONE_h_heval_$n ] && return 0; echo "no snapshot for $n"; return 1; }
  sha256sum $S/weights.pt | cut -c1-16 > $H/$n/endpoint_sha16.txt
  local T=$H/pairs_$n; mkdir -p $T/A $T/B; ln -sfn $A $T/A/state; ln -sfn $S $T/B/state
  go h_pair_$n /root/miniconda3/bin/python $W/phase2_card.py --phase3 $T --out $D/pairs_$n --device cuda --ab_only
  go h_heval_$n python $W/drift_diag.py --stage heval --out $W/diag25/f1 --deterministic --tok_mode manual --tag h_$n --states "$n=$S"
  printf '[{"name":"%s","base":"B","swap_from":null,"groups":[]}]\n' $n > $D/cfg_$n.json
  go h_segnll_$n python state_mix_eval.py --stateA $A --stateB $S --data $P --configs $D/cfg_$n.json --out $D/segnll --nll
  # 三项都完成才删快照；否则保留供重试
  if [ -f $W/status/DONE_h_pair_$n ] && [ -f $W/status/DONE_h_heval_$n ] && [ -f $W/status/DONE_h_segnll_$n ]; then rm -rf $H/$n/snaps; fi
  df -h $W | tail -1; }

cell() { train $1 $2 $3 $1_$2_s$3 && readout $1 $1_$2_s$3; }

neuron() {  # neuron <tag> <sw> <run names...>：选择性重合与 ρ（down 列、gate/up 行）
  local tag=$1 sw=$2; shift 2; local F=""
  for n in "$@"; do [ -f $D/pairs_$n/A_B_fields.npz ] && F="$F $n=$D/pairs_$n/A_B_fields.npz"; done
  [ -z "$F" ] && return 0
  case $sw in
    MG) go h_neuron_$tag python $W/neuron_act.py --state $M  --name M_$tag   --out $D --deterministic --fields $F;;
    GM) go h_neuron_$tag python $W/neuron_act.py --state $AG --name A_g_$tag --out $D --deterministic --fields $F;;   # 新任务 = MBPP，选择性符号在本地翻转（同 D 轮）
    PG) go h_neuron_$tag python $W/neuron_act.py --state $BG --name B_g_$tag --out $D --deterministic --tasks gsm8k,mbpp,dolly_qa,bio --sel gsm8k,bio --fields $F;;
  esac; }

pack() {    # pack <tag>：小文件打包回传（不含权重）
  go h_pack_$1 bash -c "cd $W; tar czf /root/autodl-tmp/h_results_$1.tgz diag25/h runs_h/*/log.jsonl runs_h/*/endpoint_sha16.txt \$(ls runs_h/*/records/match_c.pt runs_h/*/records/upd_energy.pt runs_h/*/records/probe_full_*.pt 2>/dev/null) \$(ls diag25/f1/heval_h_*.json) \$(ls logs/h*.log) status/status.tsv; ls -la /root/autodl-tmp/h_results_$1.tgz"; }

# ================================================================ H0 指纹、冒烟、确定性复跑
[ -f $W/status/H_ABORT ] && { echo "H_ABORT 存在，先查原因再删除该标记"; exit 1; }
fp() { local got=$(sha256sum $1 | cut -c1-16); [ "$got" = "$2" ] || abort "fingerprint $1 got $got want $2"; echo "fp ok $1 $got"; }
if [ ! -f $W/status/DONE_h0_fp ]; then
  fp $M/weights.pt 5d9825b68551462d; fp $AG/weights.pt 938cdf4e74780d43; fp $BG/weights.pt 70f6268bf66fd3f7
  [ -f $W/runs_e/E_vt/snaps/step0125/weights.pt ] && fp $W/runs_e/E_vt/snaps/step0125/weights.pt b20283ded2934fc1
  touch $W/status/DONE_h0_fp
fi
# 冒烟：2 步，检查 upd_energy.pt 形状
go h0_smoke bash -c "rm -rf $H/_smoke && python fork_continue.py fork --task gsm8k --train $P/gsm8k/train.jsonl --out $H/_smoke --init $BG --epochs 1 --steps 2 --lr 5e-6 --order_seed 17 --full_eval 0 --dense '' --upd_energy --tok_mode manual --deterministic $PR4 --m_op zero --v_op keep \
  && python -c \"import torch; d=torch.load('$H/_smoke/records/upd_energy.pt'); x=d['down_col']; print(x.shape, x.dtype, float(x.sum())); assert tuple(x.shape)==(2,24,4864) and bool(torch.isfinite(x).all()) and float(x.min())>=0\" && rm -rf $H/_smoke"
[ -f $W/status/DONE_h0_smoke ] || abort "smoke failed"
# 确定性复跑：PG 留 v·留 t、种子 17（= E_vt），新代码路径（无 --fields、加 --upd_energy）下终点须逐位相同
train PG vt 17 PG_vt_s17
S=$H/PG_vt_s17/snaps/step0125
if [ -f $S/weights.pt ]; then fp $S/weights.pt b20283ded2934fc1; elif [ ! -f $W/status/DONE_h_pair_PG_vt_s17 ]; then abort "rerun PG_vt_s17 produced no snapshot"; fi
readout PG PG_vt_s17     # 种子 17 也补一份逐步能量与分解收敛（读数已有，可作同口径核对）

# ================================================================ H5′ 种子 31（MG 的 vt、0 已有 K_vt_s31、K_0_s31）
for c in vfac vperm; do cell MG $c 31; done
for c in vt vfac vperm 0; do cell GM $c 31; done
for c in vt vfac vperm 0; do cell PG $c 31; done
neuron s31 MG MG_vfac_s31 MG_vperm_s31
neuron s31g GM GM_vt_s31 GM_vfac_s31 GM_vperm_s31 GM_0_s31
neuron s31p PG PG_vt_s31 PG_vfac_s31 PG_vperm_s31 PG_0_s31
pack s31

# ================================================================ H3 种子 17 中位数校准补齐（MG 两种近似、GM 行列近似）
cell MG vfac 17; cell MG vperm 17; cell GM vfac 17
neuron s17 MG MG_vfac_s17 MG_vperm_s17
neuron s17g GM GM_vfac_s17
pack s17

# ================================================================ H5″ 种子 47（12 条）
for sw in MG GM PG; do for c in vt vfac vperm 0; do cell $sw $c 47; done; done
neuron s47 MG MG_vt_s47 MG_vfac_s47 MG_vperm_s47 MG_0_s47
neuron s47g GM GM_vt_s47 GM_vfac_s47 GM_vperm_s47 GM_0_s47
neuron s47p PG PG_vt_s47 PG_vfac_s47 PG_vperm_s47 PG_0_s47
pack s47

echo "H all done $(date +%T)"; touch $W/status/DONE_h
