#!/bin/bash

echo "gpus,N,usr_htod_memcpy_time,usr_dtoh_memcpy_time,usr_kernel_time,dep_res_creation_time,dep_res_copy_time,arg_acc_time,lin_time"
for logFile in results/*; do
	#echo $logFile
	content=$(cat $logFile | tr -s ' ' | tr -s '\t')
	N=$(echo "$content" | grep -e "- N =" | cut -d' ' -f5)
	gpus=$(echo "$content" | grep -e "true num(device) =" | cut -d' ' -f5)
	gpus=${gpus:-1}
	usr_htod_memcpy_time=$(echo "$content" | grep -e "- host to device copy time" | cut -d' ' -f9)
	usr_htod_memcpy_time=${usr_htod_memcpy_time:-"N/A"}
	usr_dtoh_memcpy_time=$(echo "$content" | grep -e "- device to host copy time" | cut -d' ' -f9)
	usr_dtoh_memcpy_time=${usr_dtoh_memcpy_time:-"N/A"}
	usr_kernel_time=$(echo "$content" | grep -e "- kernel time" | cut -d' ' -f6)
	usr_kernel_time=${usr_kernel_time:-"N/A"}
	dep_res_creation_time=$(echo "$content" | grep -e "total dep res creation time" | cut -d' ' -f8)
	dep_res_copy_time=$(echo "$content" | grep -e "total dep res time" | cut -d' ' -f7)
	dep_res_creation_time=${dep_res_creation_time:-"N/A"}
	dep_res_copy_time=${dep_res_copy_time:-"N/A"}
	arg_acc_time=$(echo "$content" | grep -e "* arg access time =" | cut -d' ' -f6)
	arg_acc_time=${arg_acc_time:-"N/A"}
	lin_time=$(echo "$content" | grep -e "* linearization time =" | cut -d' ' -f5)
	lin_time=${lin_time:-"N/A"}

	echo "$gpus,$N,$usr_htod_memcpy_time,$usr_dtoh_memcpy_time,$usr_kernel_time,$dep_res_creation_time,$dep_res_copy_time,$arg_acc_time,$lin_time"
done
