#!/usr/bin/env bash

echo "O3,gpus,N,iter,usr_htod_time,usr_kernel_time,usr_dtoh_time,dep_res_creation_time,dep_res_exec_time,linearization_time"
for ngpus in {1..16}; do
	for kN in 1 2 4 8 12 16 20 24 28 32 36; do
		for iter in {1..5}; do
			N=$((1024*kN))
			if [[ $ngpus == 1 ]]; then
				FNAME="results/stencil_${ngpus}_${kN}_${iter}.txt"
				if [[ ! -e $FNAME ]]; then
					>&2 echo "Could not find file $FNAME"
				else
					O3="false"
					content=$(cat $FNAME | tr '\t' ' ' | tr -s ' ')
					usr_htod_time=$(echo "$content" | grep "** host to device copy time = " | cut -d ' ' -f 8)
					usr_kernel_time=$(echo "$content" | grep "** kernel time = " | cut -d ' ' -f 5)
					usr_dtoh_time=$(echo "$content" | grep "** device to host copy time = " | cut -d ' ' -f 8)
					dep_res_creation_time=$(echo "$content" | grep "* total dep res creation time = " | cut -d ' ' -f 9)
					dep_res_exec_time=$(echo "$content" | grep "* total dep res time = " | cut -d ' ' -f 8)
					linearization_time=$(echo "$content" | grep "* linearization time = " | cut -d ' ' -f 6)
					usr_htod_time=${usr_htod_time:="N/A"}
					usr_kernel_time=${usr_kernel_time:="N/A"}
					usr_dtoh_time=${usr_dtoh_time:="N/A"}
					dep_res_creation_time=${dep_res_creation_time:="N/A"}
					dep_res_exec_time=${dep_res_exec_time:="N/A"}
					linearization_time=${linearization_time:="N/A"}
					echo "${O3},${ngpus},${N},${iter},${usr_htod_time},${usr_kernel_time},${usr_dtoh_time},${dep_res_creation_time},${dep_res_exec_time},${linearization_time}"
				fi

			else
				FNAME="results/stencil_${ngpus}_${kN}_${iter}_O3.txt"
				if [[ ! -e $FNAME ]]; then
					>&2 echo "Could not find file $FNAME"
				else
					O3="true"
					content=$(cat $FNAME | tr '\t' ' ' | tr -s ' ')
					usr_htod_time=$(echo "$content" | grep "** host to device copy time = " | cut -d ' ' -f 8)
					usr_kernel_time=$(echo "$content" | grep "** kernel time = " | cut -d ' ' -f 5)
					usr_dtoh_time=$(echo "$content" | grep "** device to host copy time = " | cut -d ' ' -f 8)
					dep_res_creation_time=$(echo "$content" | grep "* total dep res creation time = " | cut -d ' ' -f 9)
					dep_res_exec_time=$(echo "$content" | grep "* total dep res time = " | cut -d ' ' -f 8)
					linearization_time=$(echo "$content" | grep "* linearization time = " | cut -d ' ' -f 6)
					usr_htod_time=${usr_htod_time:="N/A"}
					usr_kernel_time=${usr_kernel_time:="N/A"}
					usr_dtoh_time=${usr_dtoh_time:="N/A"}
					dep_res_creation_time=${dep_res_creation_time:="N/A"}
					dep_res_exec_time=${dep_res_exec_time:="N/A"}
					linearization_time=${linearization_time:="N/A"}
					echo "${O3},${ngpus},${N},${iter},${usr_htod_time},${usr_kernel_time},${usr_dtoh_time},${dep_res_creation_time},${dep_res_exec_time},${linearization_time}"
				fi
			fi
		done
	done
done
