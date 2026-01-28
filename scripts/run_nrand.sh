#!/bin/bash
# Script for nrand ablation study - runs experiments with different numbers of random annotators

GPU=${1:-0}
DATASET=${2:-"cifar10"}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cll

# Configuration
STRATEGY="CPE"  # Fixed strategy
TYPE="F"      # Fixed type
SEEDSET=(42)  # Seeds to run for each nrand value

# nrand values to test (adjust range as needed)
NRAND_VALUES=(2 3 4 5 6 7 8 9 10)

# Number of parallel workers (GPU processes)
PARALLEL_WORKERS=3

# Function to run experiment
run_experiment() {
    local nrand=$1
    local seed=$2
    local gpu=$3
    
    local transition_matrix="llava_noise=False-nrand=${nrand}"
    
    echo "Starting: nrand=${nrand} seed=${seed} on GPU ${gpu}"
    echo "Dataset: ${DATASET}, Strategy: ${STRATEGY}, Type: ${TYPE}"
    
    CUDA_VISIBLE_DEVICES=${gpu} python scripts/train.py \
        --do_train \
        --strategy ${STRATEGY} \
        --type ${TYPE} \
        --model ResNet18 \
        --dataset ${DATASET} \
        --batch_size 512 \
        --valid_type Accuracy \
        --transition_matrix ${transition_matrix} \
        --epoch 100 \
        --gpu 0 \
        --seed ${seed} &
}

# Compile all combinations (nrand × seed)
combinations=()
for nrand in "${NRAND_VALUES[@]}"; do
    for seed in "${SEEDSET[@]}"; do
        combinations+=("${nrand}:${seed}")
    done
done

echo "Total experiments to run: ${#combinations[@]}"
echo "Running ${PARALLEL_WORKERS} experiments in parallel"
echo "========================================"

# Process combinations in batches
batch_num=1
for ((i=0; i<${#combinations[@]}; i+=PARALLEL_WORKERS)); do
    echo ""
    echo "=== Running Batch ${batch_num} ==="
    
    # Launch up to PARALLEL_WORKERS experiments
    for ((j=0; j<PARALLEL_WORKERS && i+j<${#combinations[@]}; j++)); do
        combo="${combinations[i+j]}"
        nrand="${combo%%:*}"
        seed="${combo#*:}"
        
        run_experiment ${nrand} ${seed} ${GPU}
        sleep 10  # Wait 5 seconds before starting next run to avoid log conflicts
    done
    
    # Wait for this batch to complete
    wait
    echo "=== Batch ${batch_num} completed ==="
    ((batch_num++))
done

echo ""
echo "========================================"
echo "All nrand ablation experiments completed!"
echo "Total experiments run: ${#combinations[@]}"
