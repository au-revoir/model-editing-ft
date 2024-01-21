#!/bin/bash

#$(conda activate dpo_hf)
# Set the initial values

starting=0
limit=3
increment=1

# Variables to store cumulative values
rewrite_success_accuracy=0
paraphrase_success_accuracy=0
neighborhood_success_accuracy=0
rewrite_magnitude=0
paraphrase_magnitude=0
neighborhood_magnitude=0
num_iterations=0

while [ $starting -lt $limit ]; do
    end=$((start + increment))

    # Call the Python script with the desired parameters and capture output
    #outputs=($(python evaluate_gpt2_with_neighbor_prompts.py --data_path data/counterfact.json --start $starting --end $end))
    outputs=($(/common/home/gg676/anaconda3/envs/dpo_hf/bin/python demo.py))

    #echo"$outputs"
    #echo "$outputs[*]"

    # Extract the six output values
    output1=$(echo $outputs | awk '{print $1}')
    output2=$(echo $outputs | awk '{print $2}')
    output3=$(echo $outputs | awk '{print $3}')
    output4=$(echo $outputs | awk '{print $4}')
    output5=$(echo $outputs | awk '{print $5}')
    output6=$(echo $outputs | awk '{print $6}')

    echo "$output1"
    echo "$output2"

    # Accumulate values
    rewrite_success_accuracy=$((rewrite_success_accuracy + output1))
    paraphrase_success_accuracy=$((paraphrase_success_accuracy + output2))
    neighborhood_success_accuracy=$((neighborhood_success_accuracy + output3))
    rewrite_magnitude=$((rewrite_magnitude + output4))
    paraphrase_magnitude=$((paraphrase_magnitude + output5))
    neighborhood_magnitude=$((neighborhood_magnitude + output6))

    # Update the start value for the next iteration
    starting=$end
    num_iterations=$((num_iterations + 1))
done

# Calculate average
#avg_output1=$((rewrite_success_accuracy / num_iterations))
#avg_output2=$((paraphrase_success_accuracy / num_iterations))
#avg_output3=$((neighborhood_success_accuracy / num_iterations))
#avg_output4=$((rewrite_magnitude / num_iterations))
#avg_output5=$((paraphrase_magnitude / num_iterations))
#avg_output6=$((neighborhood_magnitude / num_iterations))

# Display the results
echo "rewrite_success_accuracy: $rewrite_success_accuracy"
echo "paraphrase_success_accuracy: $paraphrase_success_accuracy"
echo "neighborhood_success_accuracy: $neighborhood_success_accuracy"
echo "rewrite_magnitude: $rewrite_magnitude"
echo "paraphrase_magnitude: $paraphrase_magnitude"
echo "neighborhood_magnitude: $neighborhood_magnitude"

