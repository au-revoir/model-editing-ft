#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <total_gpus> <model_path> <model_name>"
    exit 1
fi

# Set the path to your Python script
python_script="python calculate_ge_rs.py"

total_examples=10000
total_gpus=$1  

# Calculate the number of examples each GPU should process
examples_per_gpu=$((total_examples / total_gpus))
remaining_examples=$((total_examples % total_gpus))

model_path=$2
model_name=$3

results_dir="results"
if [ ! -d "$results_dir" ]; then
    mkdir "$results_dir"
fi

run_dir="${results_dir}/run_0"
count=0

while [ -d "${run_dir}" ]; do
    count=$((count + 1))
    run_dir="${results_dir}/run_${count}"
done

mkdir "$run_dir"

mkdir -p "${run_dir}/log"

declare -a pids

for ((gpu_id=0; gpu_id<total_gpus; gpu_id++)); do
    # Calculate the range of examples for the current GPU
    data_start=$((gpu_id * examples_per_gpu))
    data_end=$((data_start + examples_per_gpu - 1))

    # Add the remaining examples to the last GPU
    if [ $gpu_id -eq $((total_gpus - 1)) ]; then
        data_end=$((data_end + remaining_examples))
    fi

    # Set the output path and log file for the current GPU
    output_path="${run_dir}/gpu_${gpu_id}"
    log_file="${run_dir}/log/gpu_${gpu_id}.log"

    mkdir -p "${output_path}"

    cmd="$python_script --model_path $model_path --model_name $model_name --data_start $data_start --data_end $data_end --gpu_id $gpu_id --output_path $output_path > $log_file 2>&1 &"
    eval $cmd
    
    pids+=($!)
    
    echo "GPU $gpu_id: Processing examples $data_start to $data_end. Log file: $log_file"
    sleep 60
done

echo "Processing started for all GPUs."

for pid in "${pids[@]}"; do
    wait "$pid"
done

summarize_cmd="python summarize_consistency_fluency.py --saved_result_path ${output_path}"
eval $summarize_cmd
