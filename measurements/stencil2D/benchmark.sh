#!/usr/bin/env bash

for ngpus in {1..16}; do
	for kN in 1 2 4 8 12 16 20 24 28 32 36; do
		for iter in {1..5}; do
			N=$((1024*kN))
			if [[ $ngpus == 1 ]]; then
				FNAME=stencil_${ngpus}_${kN}_${iter}.txt
				cmd="./stencil-2D-single-gpu -N $N -T 1000"
				sopts="-p mantaro --gres=gpu:$ngpus --exclusive"
				sopts="$sopts -o results/$FNAME -J g${ngpus}_n${N}"
				sbatch $sopts --wrap "$cmd"
			else
				FNAME=stencil_${ngpus}_${kN}_${iter}.txt
				cmd="./stencil-2D-multi-gpu -N $N -T 1000"
				sopts="-p mantaro --gres=gpu:$ngpus --exclusive"
				sopts="$sopts -o results/$FNAME -J g${ngpus}_n${N}"
				sbatch $sopts --wrap "$cmd"

				FNAME=stencil_${ngpus}_${kN}_${iter}_O3.txt
				sopts="-p mantaro --gres=gpu:$ngpus --exclusive"
				sopts="$sopts -o results/$FNAME -J g${ngpus}_n${N}"
				cmd="./stencil-O3-2D-multi-gpu -N $N -T 1000"
				sbatch $sopts --wrap "$cmd"
			fi
		done
	done
done
