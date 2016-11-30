#!/bin/bash

DEBUG=false
VERBOSE=false
SLURM=false

function printUsage {
	echo "Usage: $0 [options]"
	echo "Options:"
	echo "  -d    Debug mode"
	echo "  -h    Show this help message"
	echo "  -v    Be verbose"
	echo "  -s    Use slurm to schedule the jobs"
	exit 1
}

function log {
	if [ "$VERBOSE" = true ]; then
		echo $1
	fi
}

while [[ $# > 0 ]]; do
key="$1"
	case $key in
		-d|--debug)
		DEBUG=true
		;;
		-v|--verbose)
		VERBOSE=true
		;;
		-h|--help)
		printUsage
		;;
		-s|--slurm)
		SLURM=true
		shift
		;;
		*)
		echo "***ERROR: cmd line argument not known!" # unknown option
		;;
	esac
shift # past argument or value
done

T=1
#for kN in 16 48 80 112 144 176 208; do
#for kN in $(seq 16 2 336); do
for bs in 1024; do
	for N in $(seq 61440 1024 170000); do
#	for numGPUs in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
		for numGPUs in 1; do
			for iter in 1 2 3; do
				outfile="results/multi_gpu_log_N${N}_T${T}_bs${bs}_gpus${numGPUs}_iter${iter}_$(hostname).txt"
				if [[ $numGPUs == 1 ]]; then
					cmd="./n-body-single-gpu -N $N -T $T -bs $bs"
				else
					cmd="./n-body-multi-gpu -N $N -T $T -bs $bs"
				fi
				cmd+=" -fno-pu"
				hn=$(hostname)
				if [[ "$SLURM" == "true" ]]; then
					case $hn in 
						victoria)
						sopts="-p mantaro "
						;;
						creek*)
						sopts=" "
						;;
						*)
						echo "***ERROR: I don't know the slurm partition to execute on host: $hn"
						exit 1
						;;
					esac
					sopts+="--gres=gpu:${numGPUs} "
					sopts+="--exclusive "
					sopts+="-o $outfile "
					sopts+="-J nbody-N${N}-T${T}-gpus${numGPUs}-i${iter}"
					sbatch $sopts --wrap "$cmd"
					sleep 0.3
				else
					if [[ "$hn" == "victoria" || "$hn" == "octane" ]]; then
						echo "Do you want to execute your commands without slurm on $hn? (Yes|[No]):"
						read answer
						if [[ "$answer" != "Yes" ]]; then
							echo "Exit ..."
							exit 1
						fi
					fi
					echo "Going to execute: $cmd"
					$cmd > $outfile
				fi
			done
		done
	done
done
