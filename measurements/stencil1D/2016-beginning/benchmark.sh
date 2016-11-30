#!/usr/bin/env bash

for ngpus in {1..16}; do
	for kN in 1 2 4 8 12 16 20 24 28 32 36; do
		N=$((1024*kN))
		FNAME=stencil_${ngpus}gpus_${N}N.txt
		if [ $ngpus == 1 ]; then
			appendix=single-gpu
		else
			appendix=cpall
		fi
		cmd="./stencil-$appendix -N $N -T 1000"
		sopts="-p mantaro --gres=gpu:$ngpus --exclusive"
		sopts="$sopts -o results/$FNAME -J g${ngpus}_n${N}"
		sbatch $sopts --wrap "$cmd"
	done
done
#done
