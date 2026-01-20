#!/bin/bash
# this script is meant to run all algorithm variants on a specific dataset

GPU=${1:-0}
DATASET="cifar100"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cll

# # Algorithm sets with their types
# declare -A ALGOSET1=(["FWD"]="NL")
# declare -A ALGOSET2=(["URE"]="TNN TGA")
# declare -A ALGOSET3=(["CPE"]="F I T")

declare -A ALGOSET1=(["CE"]="NL")

# SEEDSET=(42 1126 2202)
SEEDSET=(42 1126 2202)
# Function to run experiment
run_experiment() {
    local strategy=$1
    local type=$2
    local seed=$3
    local gpu=$4
    
    
    echo "Starting: ${strategy} type=${type} seed=${seed} on GPU ${gpu}"
    echo "Dataset: ${DATASET}"
    
    CUDA_VISIBLE_DEVICES=${gpu} python scripts/train.py \
        --do_train \
        --strategy ${strategy} \
        --type ${type} \
        --model ResNet34 \
        --dataset ${DATASET} \
        --batch_size 512 \
        --valid_type Accuracy \
        --transition_matrix "llava_true" \
        --gpu 0 \
        --seed ${seed} &
}

# Collect all algorithm-type combinations
combinations=()

for algo in "${!ALGOSET1[@]}"; do
    for type in ${ALGOSET1[$algo]}; do
        combinations+=("${algo}:${type}")
    done
done

for algo in "${!ALGOSET2[@]}"; do
    for type in ${ALGOSET2[$algo]}; do
        combinations+=("${algo}:${type}")
    done
done

for algo in "${!ALGOSET3[@]}"; do
    for type in ${ALGOSET3[$algo]}; do
        combinations+=("${algo}:${type}")
    done
done

# Process combinations in batches of 3 (3 combinations × 3 seeds = 9 processes)
batch_size=3
batch_num=1

for ((i=0; i<${#combinations[@]}; i+=batch_size)); do
    echo "=== Running Batch ${batch_num} ==="
    
    # Get up to batch_size combinations
    for ((j=0; j<batch_size && i+j<${#combinations[@]}; j++)); do
        combo="${combinations[i+j]}"
        strategy="${combo%%:*}"
        type="${combo#*:}"
        
        echo "  Starting ${strategy} ${type} (3 seeds)"
        
        # Run all seeds for this combination
        for seed in "${SEEDSET[@]}"; do
            run_experiment "${strategy}" "${type}" ${seed} ${GPU}
            sleep 5  # Wait 5 seconds before starting next run to avoid log conflicts
        done
    done
    
    # Wait for this batch to complete
    wait
    echo "=== Batch ${batch_num} completed ==="
    ((batch_num++))
done

echo "All experiments completed!"