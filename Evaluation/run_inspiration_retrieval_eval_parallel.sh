#!/bin/bash
# Multi-node parallel evaluation script for Inspiration Retrieval
# Automatically splits files across available GPUs on all nodes
#
# Performance optimizations:
# - Flash Attention 2 for 2x faster attention
# - Batch generation for better GPU utilization
# - Multi-GPU parallel processing

set -e

#############################################################################
# Multi-Node Evaluation Script for Inspiration Retrieval
# Supports both single-node and multi-node evaluation
# 
# Usage:
#   Single-node (8 GPUs):
#     bash run_inspiration_retrieval_eval_parallel.sh
#
#   Multi-node:
#     The script uses the following environment variables (commonly set by cluster schedulers):
#     - MLP_ROLE_INDEX: Node rank (0, 1, 2, ...)
#     - MLP_WORKER_NUM: Number of nodes
#     - MLP_WORKER_GPU: GPUs per node (8)
#
#     Samples are distributed across ALL GPUs on ALL nodes.
#     Only the master node (rank 0) combines final results.
#############################################################################

# === MULTI-NODE CONFIGURATION ===
export NNODES=${MLP_WORKER_NUM:-${NNODES:-1}}
export NODE_RANK=${MLP_ROLE_INDEX:-${NODE_RANK:-0}}
export GPUS_PER_NODE=${MLP_WORKER_GPU:-${NPROC_PER_NODE:-8}}

# Detect available GPUs on this node FIRST
LOCAL_NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
if [ -z "$LOCAL_NUM_GPUS" ] || [ "$LOCAL_NUM_GPUS" -eq 0 ]; then
    LOCAL_NUM_GPUS=$GPUS_PER_NODE
fi

# For single-node: use actual detected GPUs as GPUS_PER_NODE
# For multi-node: use configured GPUS_PER_NODE
if [ "$NNODES" -eq 1 ] && [ -z "$MLP_WORKER_GPU" ]; then
    GPUS_PER_NODE=$LOCAL_NUM_GPUS
fi

export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "=== Multi-Node Evaluation Configuration (Inspiration Retrieval) ==="
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "WORLD_SIZE (total GPUs): $WORLD_SIZE"
echo "LOCAL_NUM_GPUS: $LOCAL_NUM_GPUS"
echo ""

### Parse arguments: values to change
## MODEL_PATH
# MODEL_PATH="${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-7B"  # Base model
MODEL_PATH="${IR_CHECKPOINT_DIR}"  # Fine-tuned IR model
## LORA_PATH
LORA_PATH=""
## OUTPUT_DIR
OUTPUT_DIR="${EVAL_OUTPUT_DIR}/inspiration_retrieval"

## Generation parameters
MAX_NEW_TOKENS=16384  # Match HC eval; enough for reasoning trace + selection
TEMPERATURE=0.1
TOP_P=0.95

## Batch size for better GPU utilization
## 7B model: 8-16 recommended (lower to avoid OOM with long generation)
## 32B model: 2-4 recommended
BATCH_SIZE=4

## Retry settings for failed extractions
## Set to 0 to disable retries, 2-3 recommended for production
MAX_RETRIES=2

# Parse arguments: Default values
DATA_FILE="${IR_SFT_DATA_DIR}/eval.jsonl"
OVERLAPPING_DIR=""
RANDOM_SEED=42
OTHER_ARGS=()

echo "MODEL_PATH: $MODEL_PATH"
echo "LORA_PATH: $LORA_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MAX_RETRIES: $MAX_RETRIES"
echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "TEMPERATURE: $TEMPERATURE"
echo "DATA_FILE: $DATA_FILE"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --lora_path)
            LORA_PATH="$2"
            shift 2
            ;;
        --data_file)
            DATA_FILE="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --overlapping_dir)
            OVERLAPPING_DIR="$2"
            shift 2
            ;;
        --random_seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        *)
            OTHER_ARGS+=("$1")
            shift
            ;;
    esac
done

# Check required arguments
if [ -z "$MODEL_PATH" ] || [ -z "$DATA_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 --model_path PATH --data_file PATH --output_path PATH [--lora_path PATH] [other args]"
    exit 1
fi

# Warning about overlapping_dir in parallel mode
if [ -n "$OVERLAPPING_DIR" ]; then
    echo "WARNING: --overlapping_dir is not recommended in parallel mode."
    echo "         Filtering happens inside Python, so sample distribution may be uneven."
    echo "         Consider filtering the data file beforehand for accurate parallelization."
fi

# Create output directory (shared across all nodes)
mkdir -p "$(dirname "$OUTPUT_DIR")"
TEMP_DIR="${OUTPUT_DIR%.json}_temp"

# Clean up old temp directory if exists (only master node does this for shared storage)
if [ "$NODE_RANK" -eq 0 ]; then
    if [ -d "$TEMP_DIR" ]; then
        echo "Cleaning up old temp directory: $TEMP_DIR"
        rm -rf "$TEMP_DIR"
    fi
    mkdir -p "$TEMP_DIR"
    # Create a ready marker for other nodes
    touch "${TEMP_DIR}/.master_ready"
else
    # Wait for master to prepare the temp directory
    echo "Node $NODE_RANK: Waiting for master to prepare temp directory..."
    while [ ! -f "${TEMP_DIR}/.master_ready" ]; do
        sleep 1
    done
    echo "Node $NODE_RANK: Temp directory ready"
fi

# Get total number of samples from the data file
# Note: This counts samples BEFORE any filtering (e.g., overlapping_dir).
# If filtering is needed, pre-filter the data file for accurate parallel distribution.
NUM_SAMPLES=$(python -c "import json; print(len(json.load(open('$DATA_FILE'))))")
echo "Total samples in dataset: $NUM_SAMPLES"

# Save actual world size before unsetting for torch distributed
ACTUAL_WORLD_SIZE=$WORLD_SIZE

echo "Distributing $NUM_SAMPLES samples across $ACTUAL_WORLD_SIZE GPUs"

# If only 1 GPU total, run sequentially
if [ "$ACTUAL_WORLD_SIZE" -eq 1 ]; then
    echo "Running sequentially (1 GPU)"
    # Note: env -u WORLD_SIZE prevents transformers from auto-enabling tensor parallelism
    env -u WORLD_SIZE python ${PROJECT_ROOT}/Evaluation/inspiration_retrieval_eval.py \
        --model_path "$MODEL_PATH" \
        ${LORA_PATH:+--lora_path "$LORA_PATH"} \
        --data_file "$DATA_FILE" \
        --output_path "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --max_retries "$MAX_RETRIES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --random_seed "$RANDOM_SEED" \
        ${OVERLAPPING_DIR:+--overlapping_dir "$OVERLAPPING_DIR"} \
        "${OTHER_ARGS[@]}"
    exit 0
fi

# Calculate samples per GPU (distribute evenly with remainder going to first GPUs)
SAMPLES_PER_GPU=$((NUM_SAMPLES / ACTUAL_WORLD_SIZE))
REMAINDER=$((NUM_SAMPLES % ACTUAL_WORLD_SIZE))

echo "Node $NODE_RANK: Distributing samples across $LOCAL_NUM_GPUS local GPUs"

# Start parallel processes on this node
pids=()
for local_gpu_id in $(seq 0 $((LOCAL_NUM_GPUS - 1))); do
    # Calculate global GPU ID
    global_gpu_id=$((NODE_RANK * GPUS_PER_NODE + local_gpu_id))
    
    # Calculate start and end index for this GPU
    # First $REMAINDER GPUs get one extra sample
    if [ $global_gpu_id -lt $REMAINDER ]; then
        start_idx=$((global_gpu_id * (SAMPLES_PER_GPU + 1)))
        end_idx=$((start_idx + SAMPLES_PER_GPU + 1))
    else
        start_idx=$((REMAINDER * (SAMPLES_PER_GPU + 1) + (global_gpu_id - REMAINDER) * SAMPLES_PER_GPU))
        end_idx=$((start_idx + SAMPLES_PER_GPU))
    fi
    
    if [ $start_idx -ge $NUM_SAMPLES ]; then
        echo "  GPU $local_gpu_id (global: $global_gpu_id): No samples assigned"
        continue
    fi
    
    # Output file for this GPU
    gpu_output="${TEMP_DIR}/result_node${NODE_RANK}_gpu${local_gpu_id}.json"
    echo "  GPU $local_gpu_id (global: $global_gpu_id): Processing samples [$start_idx:$end_idx]"
    
    # Note: env -u WORLD_SIZE prevents transformers from auto-enabling tensor parallelism
    CUDA_VISIBLE_DEVICES=$local_gpu_id env -u WORLD_SIZE python ${PROJECT_ROOT}/Evaluation/inspiration_retrieval_eval.py \
        --model_path "$MODEL_PATH" \
        ${LORA_PATH:+--lora_path "$LORA_PATH"} \
        --data_file "$DATA_FILE" \
        --output_path "$gpu_output" \
        --batch_size "$BATCH_SIZE" \
        --max_retries "$MAX_RETRIES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --random_seed "$RANDOM_SEED" \
        --start_idx "$start_idx" \
        --end_idx "$end_idx" \
        ${OVERLAPPING_DIR:+--overlapping_dir "$OVERLAPPING_DIR"} \
        "${OTHER_ARGS[@]}" > "${TEMP_DIR}/node${NODE_RANK}_gpu${local_gpu_id}.log" 2>&1 &
    
    pids+=($!)
    echo "    Started (PID: ${pids[-1]})"
done

# Wait for all local processes on this node
echo "Node $NODE_RANK: Waiting for ${#pids[@]} local GPU processes to complete..."
failed=0
for i in "${!pids[@]}"; do
    if wait ${pids[$i]}; then
        echo "  Local GPU $i completed successfully"
    else
        echo "  Local GPU $i failed (check ${TEMP_DIR}/node${NODE_RANK}_gpu${i}.log)"
        ((failed++))
    fi
done

if [ $failed -gt 0 ]; then
    echo "Warning: $failed local GPU(s) failed on node $NODE_RANK"
fi

# Create a marker file to indicate this node is done
touch "${TEMP_DIR}/.node${NODE_RANK}_done"
echo "Node $NODE_RANK: Marked as complete"

# Only master node (rank 0) combines results
if [ "$NODE_RANK" -eq 0 ]; then
    echo ""
    echo "=== Master Node: Waiting for all nodes to complete ==="
    
    # Wait for all nodes to complete (check for marker files)
    max_wait=3600  # 1 hour timeout
    wait_interval=10
    elapsed=0
    
    while [ $elapsed -lt $max_wait ]; do
        all_done=true
        for node_id in $(seq 0 $((NNODES - 1))); do
            if [ ! -f "${TEMP_DIR}/.node${node_id}_done" ]; then
                all_done=false
                break
            fi
        done
        
        if $all_done; then
            echo "All $NNODES nodes completed!"
            break
        fi
        
        echo "  Waiting for nodes... (${elapsed}s elapsed)"
        sleep $wait_interval
        elapsed=$((elapsed + wait_interval))
    done
    
    if [ $elapsed -ge $max_wait ]; then
        echo "WARNING: Timeout waiting for all nodes. Combining available results..."
    fi
    
    # Combine results from all nodes
    echo "Combining results from all nodes..."
    python -c "
import json
import glob
import os

temp_dir = '$TEMP_DIR'
output_path = '$OUTPUT_DIR'

# Find all GPU result files from all nodes
gpu_files = glob.glob(os.path.join(temp_dir, 'result_node*_gpu*.json'))
gpu_files = sorted(gpu_files)  # Sort for consistent ordering

if not gpu_files:
    print(f'WARNING: No GPU result files found in {temp_dir}')
    print('Check GPU logs for errors.')
    exit(1)

all_results = []
total_correct = 0
total_samples = 0
total_time = 0

# Collect data from each GPU file
for gpu_file in gpu_files:
    try:
        with open(gpu_file) as f:
            data = json.load(f)
            
        if 'individual_results' in data:
            all_results.extend(data['individual_results'])
        
        if 'overall_metrics' in data:
            metrics = data['overall_metrics']
            total_correct += metrics.get('correct_predictions', 0)
            total_samples += metrics.get('total_samples', 0)
            total_time += metrics.get('total_time', 0)
            print(f'  Loaded {gpu_file}: {metrics.get(\"total_samples\", 0)} samples, {metrics.get(\"accuracy\", 0):.2%} accuracy')
        else:
            print(f'  Loaded {gpu_file}: (no metrics found, only individual_results)')
            
    except Exception as e:
        print(f'Error reading {gpu_file}: {e}')

# Calculate combined metrics
accuracy = total_correct / total_samples if total_samples > 0 else 0.0
avg_time = total_time / total_samples if total_samples > 0 else 0.0

combined_results = {
    'overall_metrics': {
        'total_samples': total_samples,
        'correct_predictions': total_correct,
        'accuracy': accuracy,
        'average_generation_time': avg_time,
        'total_time': total_time,
        'num_nodes': $NNODES,
        'total_gpus': $ACTUAL_WORLD_SIZE
    },
    'individual_results': all_results
}

# Save combined results
with open(output_path, 'w') as f:
    json.dump(combined_results, f, indent=2)

print(f'')
print(f'Combined {len(all_results)} results from {len(gpu_files)} GPU processes')
print(f'Total samples: {total_samples}')
print(f'Correct predictions: {total_correct}')
print(f'Accuracy: {accuracy:.2%}')
print(f'Total time: {total_time:.1f}s')
print(f'Throughput: {total_samples / total_time:.1f} samples/sec' if total_time > 0 else 'N/A')
print(f'Results saved to {output_path}')
"

    # Cleanup temp directory (only on master)
    rm -rf "$TEMP_DIR"
    
    echo ""
    echo "=== Evaluation complete. Results saved to $OUTPUT_DIR ==="
else
    echo "Node $NODE_RANK: Finished. Master node will combine results."
fi
