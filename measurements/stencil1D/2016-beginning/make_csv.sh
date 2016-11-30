#!/bin/bash

echo 'N,gpus,htod,kernel,dtoh,T,depResTime,depResSize[MB]'

for ngpus in {1..16}; do
	for kN in 1 2 4 8 12 16 20 24 28 32 36; do
		N=$((1024*kN))
		FNAME=stencil_${ngpus}gpus_${N}N.txt
		htod=$(grep '  host to device copy time' results/$FNAME | cut -d ' ' -f9)
		dtoh=$(grep '  device to host copy time' results/$FNAME | cut -d ' ' -f9)
		kernel=$(grep '  kernel time' results/$FNAME | cut -d ' ' -f6)
		T=$(grep 'T[[:space:]]*=' results/$FNAME | cut -d ' ' -f8)
		depResTime=$(grep 'total dep res time' results/$FNAME | cut -d ' ' -f7)
		if [ -z $depResTime ]; then
			depResTime=N/A
		else
			depResTime=${depResTime%s}
		fi
		depResSize=$(grep 'total dep res memcpy size' results/$FNAME | cut -d ' ' -f8)
		if [ -z $depResSize ]; then
			depResSize=N/A
		else
			depResSize=${depResSize%MB}
		fi
		if [ -z "$kernel" -o -z "$htod" -o -z "$dtoh" ]; then
			echo $N,$ngpus,N/A,N/A,N/A,N/A
		else
			echo $N,$ngpus,$htod,$kernel,$dtoh,$T,$depResTime,$depResSize
		fi
	done
done
