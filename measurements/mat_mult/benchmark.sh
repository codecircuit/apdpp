#!/usr/bin/env bash

bs=32
for kN in 1 2 4 8 12 16 20 24 28; do
	N=$((kN*1024))
	for numGPUs in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
		for iter in {1..5}; do
			outfile="results/mat-mul_N${N}_bs${bs}_gpus${numGPUs}_iter${iter}.txt"
			if [[ $numGPUs == 2 ]]; then
				cmd="source /home/amatz/victoria-guard.sh; ./mat-mul-single-gpu -N $N -bs $bs"
			else
				cmd="source /home/amatz/victoria-guard.sh; ./mat-mul-multi-gpu -N $N -bs $bs"
			fi
			sbatch --gres=gpu:${numGPUs} --exclusive -p mantaro -o $outfile -J "mat-mul-N${N}-gpus${numGPUs}-i${iter}" --wrap "$cmd"
			sleep 0.2
		done
	done
done
