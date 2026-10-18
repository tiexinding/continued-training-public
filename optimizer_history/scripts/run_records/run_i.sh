# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# I 轮（文档 43，10-10 开卡）GPU 串行队列：
#   i_test  MG_vfac_s17 带只读钩子重跑，终点 sha 必须等于 H 轮记录（57f391d583e3a799），否则 I_ABORT；
#   I1      8 条 calib_check --applied（与表 9 同 8 条）；
#   I2      其余 8 条种子 17 重跑（三切换 × 留 v·t / 行列近似 / 置换）+ --upd_stats，逐条核对与旧运行逐位一致后删快照。
# 磁盘：卡上约 6 G 空余，同一时刻只留一份快照（2 G）。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets
M=$W/runs_t1/lr1e-5/M/state; AG=$W/runs_d/A_g/state; BG=$W/runs_e/B_g/state
I=$W/runs_i; D=$W/diag25/i
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/fork; mkdir -p $I $D/calib $W/status $W/logs
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
PR3="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl"
PR4="$PR3 --probe bio=$P/bio/probe.jsonl"
sw_init()  { case $1 in MG) echo $M;; GM) echo $AG;; PG) echo $BG;; esac; }
sw_task()  { case $1 in MG|PG) echo gsm8k;; GM) echo mbpp;; esac; }
sw_steps() { case $1 in MG|PG) echo 125;; GM) echo 108;; esac; }
sw_pr()    { case $1 in PG) echo "$PR4";; *) echo "$PR3";; esac; }
cond_op()  { case $1 in vt) echo "--m_op zero --v_op keep";; vfac) echo "--m_op zero --v_op ladder:rowcol:matched_med";;
                        vperm) echo "--m_op zero --v_op ladder:cperm:matched_med";; esac; }

# 逐位核对：ref = sha16:<值> | snap:<旧快照 weights.pt> | log:<旧 log.jsonl>（比最后一次完整探针评估）
verify() { local n=$1 ref=$2; local S=$I/$n/snaps/step$(printf %04d $3)/weights.pt; local new=$(sha256sum $S | cut -c1-16); local ok=0 old=""
  case $ref in
    sha16:*) old=${ref#sha16:}; [ "$new" = "$old" ] && ok=1;;
    snap:*)  old=$(sha256sum ${ref#snap:} | cut -c1-16); [ "$new" = "$old" ] && ok=1;;
    log:*)   old=$(python - "$I/$n/log.jsonl" "${ref#log:}" <<'PY'
import json, sys
def last(p): return [json.loads(l) for l in open(p) if '"eval_full"' in l][-1]
a, b = last(sys.argv[1]), last(sys.argv[2]); ks = [k for k in a if k not in ("event", "step")]
print("EQ" if a["step"] == b["step"] and all(a[k] == b.get(k) for k in ks) else f"NE {a} {b}")
PY
); [ "$old" = "EQ" ] && ok=1;;
  esac
  echo -e "$n\t$new\t$ref\t$old\t$ok" >> $W/status/verify_i.tsv; echo "$new" > $I/$n/endpoint_sha16.txt
  rm -rf $I/$n/snaps; return $(( 1 - ok )); }

run2() {  # run2 <sw> <cond> <ref>
  local sw=$1 cd=$2 ref=$3; local n=${sw}_${cd}_s17 st=$(sw_steps $1)
  [ -f $W/status/DONE_i_train_$n ] || rm -rf $I/$n
  go i_train_$n python fork_continue.py fork --task $(sw_task $sw) --train $P/$(sw_task $sw)/train.jsonl --out $I/$n --init $(sw_init $sw) \
     --epochs 1 --steps $st --lr 5e-6 --order_seed 17 --full_eval 0,25,50,75,100,$st --dense "" --snap $st --upd_energy \
     --upd_stats --upd_rows $W/token_groups_592.json --tok_mode manual --deterministic $(sw_pr $sw) $(cond_op $cd) || return 1
  [ -f $W/status/DONE_i_verify_$n ] && return 0
  verify $n $ref $st && touch $W/status/DONE_i_verify_$n; }

ck() { local n=$1 init=$2 task=$3 vop=$4 mc=$5; shift 5
  go i1_$n python $W/calib_check.py --init $init --task $task --v_op $vop --order_seed 17 --match_c $mc --out $D/calib/$n.json --applied --rows $W/token_groups_592.json "$@"; }

[ -f $W/status/I_ABORT ] && { echo "I_ABORT 存在，先查原因"; exit 1; }
# ---- 测试兼正式：钩子不改训练
run2 MG vfac sha16:57f391d583e3a799 || { echo "i_test 逐位核对失败"; touch $W/status/I_ABORT; exit 1; }
touch $W/status/DONE_i_test
# ---- I1
ck GM_vfac_rms   $AG mbpp  ladder:rowcol:matched     $W/runs_d/R_vfac/records/match_c.pt
ck GM_vperm_rms  $AG mbpp  ladder:cperm:matched      $W/runs_d/R_vperm/records/match_c.pt
ck GM_vperm_med  $AG mbpp  ladder:cperm:matched_med  $W/runs_e/R_vperm_med/records/match_c.pt
ck GM_vfac_med   $AG mbpp  ladder:rowcol:matched_med $W/runs_h/GM_vfac_s17/records/match_c.pt
ck PG_vfac_med   $BG gsm8k ladder:rowcol:matched_med $W/runs_e/E_vfac/records/match_c.pt
ck PG_vperm_med  $BG gsm8k ladder:cperm:matched_med  $W/runs_e/E_vperm/records/match_c.pt
ck MG_vfac_med   $M  gsm8k ladder:rowcol:matched_med $W/runs_h/MG_vfac_s17/records/match_c.pt
ck MG_vperm_med  $M  gsm8k ladder:cperm:matched_med  $W/runs_h/MG_vperm_s17/records/match_c.pt
touch $W/status/DONE_i1
# ---- I2（其余 8 条）
run2 MG vperm sha16:fe9a49b28d2721f4
run2 MG vt    log:$W/runs_b2/Kv/log.jsonl
run2 GM vfac  sha16:b72414556ec8f697
run2 GM vperm log:$W/runs_e/R_vperm_med/log.jsonl
run2 GM vt    log:$W/runs_d/R_vt/log.jsonl
run2 PG vt    sha16:b20283ded2934fc1
run2 PG vfac  log:$W/runs_e/E_vfac/log.jsonl
run2 PG vperm snap:$W/runs_e/E_vperm/snaps/step0125/weights.pt
touch $W/status/DONE_i2
go i_pack bash -c "cd $W; tar czf /root/autodl-tmp/i_results.tgz diag25/i runs_i/*/log.jsonl runs_i/*/endpoint_sha16.txt runs_i/*/records/upd_stats.pt runs_i/*/records/match_c.pt logs/i*.log status/status.tsv status/verify_i.tsv; ls -la /root/autodl-tmp/i_results.tgz"
touch $W/status/DONE_i; echo "I GPU queue done $(date +%T)"
