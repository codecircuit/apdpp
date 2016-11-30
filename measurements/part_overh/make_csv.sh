#!/usr/bin/env bash

echo "kernels,time[s]"
for i in {1..512}; do
	kN=$((16*i))
	FNAME=results/stencil_mult_${kN}.txt
	time=$(cat $FNAME | grep "kernel time" | cut -d ' ' -f6)
	echo ${kN},${time}
done
