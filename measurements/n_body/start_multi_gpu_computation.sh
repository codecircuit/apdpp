#!/bin/bash

bs=64
T=1000
for kN in 1 10 20 30 40 50; do
	N=$((kN * 1024))
	for numGPUs in 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
		cmd="source /home/amatz/victoria-guard.sh; ./n-body-multi-gpu -N $N -T $T -bs $bs -fcheck-file single_gpu_results/single_gpu_res_N${N}_T${T}_bs${bs}.csv -fkernel kernel.ptx"
		sbatch --gres=gpu:$((numGPUs + 1)) --exclusive -p mantaro -o multi_gpu_results/multi_gpu_log_N${N}_T${T}_bs${bs}_gpus${numGPUs}.txt -J nbody-${numGPUs}gpus-${N} --wrap "$cmd"
	done
done
