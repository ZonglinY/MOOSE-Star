#!/bin/bash
# Multi-node parallel evaluation script for BOUNDED hypothesis composition (Rubric Scoring)
# Automatically splits files across available GPUs on all nodes
#
# Uses bounded_selections_dir for bounded composition evaluation
# Evaluates across different tiers (hard, medium, easy)
# Reports tier-wise statistics to understand robustness to retrieval quality

set -e

#############################################################################
# Multi-Node Evaluation Script for BOUNDED Hypothesis Composition
#
# Usage:
#   Single-node (8 GPUs):
#     bash run_hypothesis_composition_eval_rubric_bounded_parallel.sh
#
#   Multi-node:
#     The script uses the following environment variables (commonly set by cluster schedulers):
#     - MLP_ROLE_INDEX: Node rank (0, 1, 2, ...)
#     - MLP_WORKER_NUM: Number of nodes
#     - MLP_WORKER_GPU: GPUs per node (8)
#############################################################################

# === MULTI-NODE CONFIGURATION ===
export NNODES=${MLP_WORKER_NUM:-${NNODES:-1}}
export NODE_RANK=${MLP_ROLE_INDEX:-${NODE_RANK:-0}}
export GPUS_PER_NODE=${MLP_WORKER_GPU:-${NPROC_PER_NODE:-8}}
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# Detect available GPUs on this node
LOCAL_NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
if [ -z "$LOCAL_NUM_GPUS" ] || [ "$LOCAL_NUM_GPUS" -eq 0 ]; then
    LOCAL_NUM_GPUS=$GPUS_PER_NODE
fi

echo "=== Multi-Node Evaluation Configuration (Bounded Composition) ==="
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "WORLD_SIZE (total GPUs): $WORLD_SIZE"
echo "LOCAL_NUM_GPUS: $LOCAL_NUM_GPUS"
echo ""

### Parse arguments: values to change
## MODEL_PATH
# MODEL_PATH="${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-7B"  # Base model
MODEL_PATH="${HC_CHECKPOINT_DIR}"  # Fine-tuned HC model
## LORA_PATH
LORA_PATH=""
## OUTPUT_DIR
OUTPUT_DIR="${EVAL_OUTPUT_DIR}/hypothesis_composition_bounded"

## Bounded selections directory (input)
BOUNDED_SELECTIONS_DIR="${BOUNDED_SELECTIONS_DIR}"

## Tiers to evaluate (comma-separated)
TIERS="hard,medium,easy"

## API max tokens for rubric evaluation output
API_MAX_TOKENS=32768

## Generation parameters
MAX_NEW_TOKENS=8192
TEMPERATURE=0.6
TOP_P=0.9
REPETITION_PENALTY=1.2

## Batch size for GPU utilization
## Larger = better GPU util, but more VRAM
## 7B model: 32-64 recommended
## 32B model: 8-16 recommended
BATCH_SIZE=48

## Parallel LLM evaluation workers (API calls)
## Should be >= batch_size to avoid eval becoming bottleneck
EVAL_MAX_WORKERS=96

# API settings
MODEL_NAME="R1-Distill-Qwen-32B"
API_TYPE=0
API_KEY="YOUR_API_KEY_HERE"
BASE_URL="${API_BASE_URL}"
OTHER_ARGS=()

echo "MODEL_PATH: $MODEL_PATH"
echo "LORA_PATH: $LORA_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "BOUNDED_SELECTIONS_DIR: $BOUNDED_SELECTIONS_DIR"
echo "TIERS: $TIERS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo ""

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
        --bounded_selections_dir)
            BOUNDED_SELECTIONS_DIR="$2"
            shift 2
            ;;
        --tiers)
            TIERS="$2"
            shift 2
            ;;
        --eval_result_path)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --api_type)
            API_TYPE="$2"
            shift 2
            ;;
        --api_key)
            API_KEY="$2"
            shift 2
            ;;
        --base_url)
            BASE_URL="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --eval_max_workers)
            EVAL_MAX_WORKERS="$2"
            shift 2
            ;;
        *)
            OTHER_ARGS+=("$1")
            shift
            ;;
    esac
done

# Check required arguments
if [ -z "$MODEL_PATH" ] || [ -z "$BOUNDED_SELECTIONS_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 --model_path PATH --bounded_selections_dir PATH --eval_result_path PATH [--lora_path PATH] [other args]"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_DIR")"
TEMP_DIR="${OUTPUT_DIR%.json}_temp"
mkdir -p "$TEMP_DIR"

# Get list of JSON files
eval_files=($(ls "$BOUNDED_SELECTIONS_DIR"/*.json 2>/dev/null | xargs -n1 basename | sort))
num_files=${#eval_files[@]}

if [ $num_files -eq 0 ]; then
    echo "No JSON files found in $BOUNDED_SELECTIONS_DIR"
    exit 1
fi

# Save actual world size before overriding for torch distributed
ACTUAL_WORLD_SIZE=$WORLD_SIZE

echo "Found $num_files files to process across $ACTUAL_WORLD_SIZE GPUs"

# Disable tensor parallelism for data parallel evaluation
# UNSET distributed env vars to prevent torch from trying to init distributed
unset RANK
unset WORLD_SIZE
unset LOCAL_RANK
unset MASTER_ADDR
unset MASTER_PORT

# If only 1 GPU total, run sequentially
if [ "$ACTUAL_WORLD_SIZE" -eq 1 ] || [ $num_files -eq 1 ]; then
    echo "Running sequentially (1 GPU or 1 file)"
    python ${PROJECT_ROOT}/Evaluation/hypothesis_composition_eval_rubric_bounded.py \
        --model_path "$MODEL_PATH" \
        ${LORA_PATH:+--lora_path "$LORA_PATH"} \
        --bounded_selections_dir "$BOUNDED_SELECTIONS_DIR" \
        --tiers "$TIERS" \
        --eval_result_path "$OUTPUT_DIR" \
        ${MODEL_NAME:+--model_name "$MODEL_NAME"} \
        ${API_TYPE:+--api_type "$API_TYPE"} \
        ${API_KEY:+--api_key "$API_KEY"} \
        ${BASE_URL:+--base_url "$BASE_URL"} \
        --batch_size "$BATCH_SIZE" \
        --eval_max_workers "$EVAL_MAX_WORKERS" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --repetition_penalty "$REPETITION_PENALTY" \
        --api_max_tokens "$API_MAX_TOKENS" \
        "${OTHER_ARGS[@]}"
    exit 0
fi

# Calculate which files this node should process
echo "Node $NODE_RANK: Distributing files across $LOCAL_NUM_GPUS local GPUs"

# Start parallel processes on this node
pids=()
for local_gpu_id in $(seq 0 $((LOCAL_NUM_GPUS - 1))); do
    global_gpu_id=$((NODE_RANK * GPUS_PER_NODE + local_gpu_id))
    
    # Collect files for this global GPU
    gpu_files=()
    for i in $(seq 0 $((num_files - 1))); do
        if [ $((i % ACTUAL_WORLD_SIZE)) -eq $global_gpu_id ]; then
            gpu_files+=("${eval_files[$i]}")
        fi
    done
    
    if [ ${#gpu_files[@]} -eq 0 ]; then
        echo "  GPU $local_gpu_id (global: $global_gpu_id): No files assigned"
        continue
    fi
    
    # Create GPU-specific directory with only this GPU's files
    gpu_bounded_dir="${TEMP_DIR}/bounded_node${NODE_RANK}_gpu${local_gpu_id}"
    mkdir -p "$gpu_bounded_dir"
    
    for file in "${gpu_files[@]}"; do
        cp "$BOUNDED_SELECTIONS_DIR/$file" "$gpu_bounded_dir/"
    done
    
    # Run evaluation on this GPU
    gpu_output="${TEMP_DIR}/result_node${NODE_RANK}_gpu${local_gpu_id}"
    echo "  GPU $local_gpu_id (global: $global_gpu_id): Processing ${#gpu_files[@]} files"
    
    # Run without distributed env vars to prevent torch.distributed init
    CUDA_VISIBLE_DEVICES=$local_gpu_id \
    python ${PROJECT_ROOT}/Evaluation/hypothesis_composition_eval_rubric_bounded.py \
        --model_path "$MODEL_PATH" \
        ${LORA_PATH:+--lora_path "$LORA_PATH"} \
        --bounded_selections_dir "$gpu_bounded_dir" \
        --tiers "$TIERS" \
        --eval_result_path "$gpu_output" \
        ${MODEL_NAME:+--model_name "$MODEL_NAME"} \
        ${API_TYPE:+--api_type "$API_TYPE"} \
        ${API_KEY:+--api_key "$API_KEY"} \
        ${BASE_URL:+--base_url "$BASE_URL"} \
        --batch_size "$BATCH_SIZE" \
        --eval_max_workers "$EVAL_MAX_WORKERS" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --repetition_penalty "$REPETITION_PENALTY" \
        --api_max_tokens "$API_MAX_TOKENS" \
        "${OTHER_ARGS[@]}" > "${TEMP_DIR}/node${NODE_RANK}_gpu${local_gpu_id}.log" 2>&1 &
    
    pids+=($!)
    echo "    Started (PID: ${pids[-1]})"
done

# Wait for all local processes
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

# Mark this node as done
touch "${TEMP_DIR}/.node${NODE_RANK}_done"
echo "Node $NODE_RANK: Marked as complete"

# Only master node combines results
if [ "$NODE_RANK" -eq 0 ]; then
    echo ""
    echo "=== Master Node: Waiting for all nodes to complete ==="
    
    max_wait=3600
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
    
    # Combine results from all nodes (BOUNDED COMPOSITION VERSION)
    echo "Combining results from all nodes..."
    python -c "
import json
import glob
import os
from collections import defaultdict

temp_dir = '$TEMP_DIR'
output_folder = '$OUTPUT_DIR'.rstrip('.json')
tiers_str = '$TIERS'
tiers = [t.strip() for t in tiers_str.split(',')]

os.makedirs(output_folder, exist_ok=True)

# Find all GPU result folders
gpu_folders = glob.glob(os.path.join(temp_dir, 'result_node*_gpu*'))
gpu_folders = [f for f in gpu_folders if os.path.isdir(f)]

if not gpu_folders:
    print(f'WARNING: No GPU result folders found in {temp_dir}')

all_metrics = []
all_generations = []

# Collect data from each GPU folder
for gpu_folder in sorted(gpu_folders):
    metrics_path = os.path.join(gpu_folder, 'metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_metrics.extend(data)
        except Exception as e:
            print(f'Error reading {metrics_path}: {e}')
    
    gen_path = os.path.join(gpu_folder, 'generations.json')
    if os.path.exists(gen_path):
        try:
            with open(gen_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_generations.extend(data)
        except Exception as e:
            print(f'Error reading {gen_path}: {e}')

# Save combined metrics
with open(os.path.join(output_folder, 'metrics.json'), 'w') as f:
    json.dump(all_metrics, f, indent=2)

# Save combined generations
with open(os.path.join(output_folder, 'generations.json'), 'w') as f:
    json.dump(all_generations, f, indent=2)

# Calculate tier-wise statistics
tier_results = {tier: [] for tier in tiers}
for m in all_metrics:
    tier = m.get('tier')
    if tier in tier_results and m.get('total_score') is not None:
        tier_results[tier].append(m)

tier_metrics = {}
all_scores = []

for tier in tiers:
    results = tier_results[tier]
    if results:
        total_scores = [r['total_score'] for r in results]
        motivation_scores = [r['scores']['motivation'] for r in results]
        mechanism_scores = [r['scores']['mechanism'] for r in results]
        methodology_scores = [r['scores']['methodology'] for r in results]
        
        tier_metrics[tier] = {
            'mean_total_score': sum(total_scores) / len(total_scores),
            'mean_motivation': sum(motivation_scores) / len(motivation_scores),
            'mean_mechanism': sum(mechanism_scores) / len(mechanism_scores),
            'mean_methodology': sum(methodology_scores) / len(methodology_scores),
            'count': len(results),
            'min_score': min(total_scores),
            'max_score': max(total_scores)
        }
        all_scores.extend(total_scores)
    else:
        tier_metrics[tier] = None

# Calculate extraction failures
failed_count = sum(1 for g in all_generations if g.get('extraction_failed', False))

# Save summary
summary = {
    'overall_mean_total_score': sum(all_scores) / len(all_scores) if all_scores else None,
    'total_evaluations': len(all_scores),
    'total_samples_processed': len(all_generations),
    'extraction_failures': failed_count,
    'tier_metrics': tier_metrics,
    'tiers_evaluated': tiers,
    'num_nodes': $NNODES,
    'total_gpus': $ACTUAL_WORLD_SIZE
}

with open(os.path.join(output_folder, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary
print(f'Combined {len(all_metrics)} results from {len(gpu_folders)} GPU processes into {output_folder}')
print(f'')
print('=== Tier-wise Results ===')
print(f'{\"Tier\":<10} {\"Count\":>8} {\"Mean Score\":>12} {\"Motivation\":>12} {\"Mechanism\":>12} {\"Methodology\":>12}')
print('-' * 70)
for tier in tiers:
    m = tier_metrics.get(tier)
    if m:
        print(f'{tier:<10} {m[\"count\"]:>8} {m[\"mean_total_score\"]:>12.2f} {m[\"mean_motivation\"]:>12.2f} {m[\"mean_mechanism\"]:>12.2f} {m[\"mean_methodology\"]:>12.2f}')
    else:
        print(f'{tier:<10} {\"N/A\":>8} {\"N/A\":>12} {\"N/A\":>12} {\"N/A\":>12} {\"N/A\":>12}')
print('-' * 70)
if all_scores:
    print(f'Overall Mean: {sum(all_scores)/len(all_scores):.2f}/12 ({len(all_scores)} evaluations)')
"

    # Cleanup temp directory
    rm -rf "$TEMP_DIR"
    
    echo ""
    echo "=== Bounded Composition Evaluation Complete. Results saved to $OUTPUT_DIR ==="
else
    echo "Node $NODE_RANK: Finished. Master node will combine results."
fi

