#!/bin/bash
data_path="7500_counterfact.json"
model_name="gpt2-xl"
prompt_rewrite="--prompt_rewrite"
prompt_paraphrase_type="--prompt_paraphrase_type generated_prepended_examples"
prompt_neighborhood="--prompt_neighborhood"
prompt_neighborhood_type="--prompt_neighborhood_type similar_examples"
dataset_name="--dataset_name counterfact"
generated_prepended_words_path="generated_prepended_words_paraphrases.json"
num_epochs="50"
lr="2.2e-5"
gpu_prefix="--gpu_id"
script_path="python run.py"

total_dataset_size="7500"

#Change this depending on num of available GPUs
num_gpus=8

examples_per_gpu=$((total_dataset_size / num_gpus))
remaining_examples=$((total_dataset_size % num_gpus))
echo "Remaining examples $remaining_examples"

log_folder="logs"
base_output_folder="results/"

mkdir -p "$log_folder"

run_number=0
while [ -d "${base_output_folder}_${lr}_${num_epochs}_${run_number}"]; do
    ((run_number++))
done

output_save_path="${base_output_folder}_${lr}_${num_epochs}_${run_number}"

mkdir -p "$output_save_path"

for ((gpu_id = 0; gpu_id < num_gpus; gpu_id++)); do
    data_start=$((gpu_id * examples_per_gpu))
    data_end=$((data_start + examples_per_gpu))

    #Give additional examples to the last GPU
    if ((gpu_id == num_gpus - 1)); then
        data_end=$((data_end + remaining_examples))
    fi

    log_file="$log_folder/gpu_${gpu_id}_log.txt"

    cmd="$script_path $prompt_rewrite $prompt_paraphrase_type $prompt_neighborhood $prompt_neighborhood_type $dataset_name --data_path $data_path --model_name $model_name --generated_prepended_words_path $generated_prepended_words_path --num_epochs $num_epochs --lr $lr --data_start $data_start --data_end $data_end --output_save_path $output_save_path $gpu_prefix $gpu_id > $log_file 2>&1 &"
    
    echo "Running command for GPU $gpu_id: $cmd"
    eval $cmd
    
done

#Wait for all background processes to finishh
wait
echo "Results saved path: $output_save_path"
