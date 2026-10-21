# 发布包：卡上运行队列的原始记录，仅供审计，不能直接运行（见本目录 README.md）；部分路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2），仍含卡上路径与清理命令
#!/bin/bash
# F9 定位：逐矩阵撤回 → 首位置缩进概率；通道级一阶归因与 v 对照（E_vperm、E_vt、E_vfac，及 F7 的 E_v / E_vclamp 若已存在）
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; E=$W/runs_e; BG=$E/B_g/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/fork
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }
until [ -f $W/status/DONE_f2 ]; do sleep 30; done
ST="E_vperm=$E/E_vperm/snaps/step0125,E_vt=$E/E_vt/snaps/step0125,E_vfac=$E/E_vfac/snaps/step0125"
[ -f $E/E_v/snaps/step0125/weights.pt ] && ST="$ST,E_v=$E/E_v/snaps/step0125"
[ -f $E/E_vclamp/snaps/step0125/weights.pt ] && ST="$ST,E_vclamp=$E/E_vclamp/snaps/step0125"
go f9_localize python $W/drift_diag.py --stage localize --deterministic --tok_mode manual --states "$ST"
go f9_fin bash -c "cd $W; tar czf /root/autodl-tmp/f9_results.tgz diag25/f1/localize.json logs/f9*.log; ls -la /root/autodl-tmp/f9_results.tgz"
touch $W/status/DONE_f3
