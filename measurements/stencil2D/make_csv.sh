#!/usr/bin/env bash

echo "gpus,N,usr_htod_time,usr_kernel_time,usr_dtoh_time,dep_res_creation_time,dep_res_exec_time,linearization_time"
for file in $(ls results); do
	content=$(cat results/$file | tr '\t' ' ' | tr -s ' ' | sed 's/^ //')
	ngpus=$(echo "$content" | grep "true num(device) = " | cut -d ' ' -f 5)
	N=$(echo "$content" | grep "\*[[:space:]]*N[[:space:]]*=" | cut -d ' ' -f 4)
	usr_htod_time=$(echo "$content" | grep "** host to device copy time = " | cut -d ' ' -f 8)
	usr_kernel_time=$(echo "$content" | grep "** kernel time = " | cut -d ' ' -f 5)
	usr_dtoh_time=$(echo "$content" | grep "** device to host copy time = " | cut -d ' ' -f 8)
	dep_res_creation_time=$(echo "$content" | grep "* total dep res creation time = " | cut -d ' ' -f 8)
	dep_res_exec_time=$(echo "$content" | grep "* total dep res time = " | cut -d ' ' -f 7)
	linearization_time=$(echo "$content" | grep "* linearization time = " | cut -d ' ' -f 5)
	ngpus=${ngpus:="1"}
	N=${N:="N/A"}
	usr_htod_time=${usr_htod_time:="N/A"}
	usr_kernel_time=${usr_kernel_time:="N/A"}
	usr_dtoh_time=${usr_dtoh_time:="N/A"}
	dep_res_creation_time=${dep_res_creation_time:="N/A"}
	dep_res_exec_time=${dep_res_exec_time:="N/A"}
	linearization_time=${linearization_time:="N/A"}
	output+="${ngpus},${N},${usr_htod_time},${usr_kernel_time},${usr_dtoh_time},${dep_res_creation_time},${dep_res_exec_time},${linearization_time}\n"
done

printf "$output" | sort -n
