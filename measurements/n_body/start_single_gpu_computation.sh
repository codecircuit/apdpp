#!/bin/bash

bs=64
T=1000
for kN in 1 10 20 30 40 50; do  #60 70 80 90 100; do
	N=$((kN * 1024))
	cmd="source /home/amatz/victoria-guard.sh; ./n-body-single-gpu -N $N -T $T -bs $bs -fout single_gpu_results/single_gpu_res_N${N}_T${T}_bs${bs}.csv --write-only-last -fkernel kernel.ptx"
	sbatch --gres=gpu:2 --exclusive -p mantaro -o single_gpu_results/single_gpu_log_N${N}_T${T}_bs${bs}.txt -J nbody-1gpu-${N} --wrap "$cmd"
done
