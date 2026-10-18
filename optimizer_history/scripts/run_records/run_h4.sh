# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# H 轮最终汇总（卡上）：等 H3 补算结束后跑 analyze_h.py，打包处理后的结果（小文件）
W=$OH_RUNS; cd $W
while [ ! -f $W/status/DONE_h3 ] && [ ! -f $W/status/H_ABORT ]; do sleep 60; done
nice -n 19 python analyze_h.py --res h_analysis/results_old --h $W --he /tmp/humaneval.parquet --out h_analysis/out > h_analysis/out/analyze.log 2>&1
echo -e "h4_analyze\t$?\t0\t$(date +%T)" >> status/status.tsv
tar czf /root/autodl-tmp/h_final_processed.tgz h_analysis/out diag25/h/calib diag25/h/h2p_energy.json diag25/h/neuron_act_*_hall.json status/status.tsv
touch status/DONE_h4; echo "H4 done $(date +%T)"
