#!/usr/bin/env bash

for i in {1..512}; do
	kN=$((16*i))
	FNAME=stencil_mult_${kN}.txt
	cmd="./stencilMult -N 4096 -T 1000 -K $kN"
	sopts="-p mantaro --gres=gpu:1 --exclusive"
	sopts="$sopts -o results/$FNAME -J stencil_mult_$kN"
	sbatch $sopts --wrap "$cmd"
done
