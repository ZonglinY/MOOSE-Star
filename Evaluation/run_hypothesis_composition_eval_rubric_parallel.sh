#!/bin/bash
# Multi-node parallel evaluation script for hypothesis composition (Rubric Scoring)
# Automatically splits files across available GPUs on all nodes
#
# Uses rubric scoring (3 dimensions: Motivation, Mechanism, Methodology)
# Single LLM call per sample; score range: 0-12
# Added --eval_max_workers for parallel LLM evaluation

set -e

#############################################################################
# Multi-Node Evaluation Script for Hypothesis Composition (Rubric Scoring)
# Supports both single-node and multi-node evaluation
# 
# Usage:
#   Single-node (8 GPUs):
#     bash run_hypothesis_composition_eval_rubric_parallel.sh
#
#   Multi-node:
#     The script uses the following environment variables (commonly set by cluster schedulers):
#     - MLP_ROLE_INDEX: Node rank (0, 1, 2, ...)
#     - MLP_WORKER_NUM: Number of nodes
#     - MLP_WORKER_GPU: GPUs per node (8)
#
#     Files are distributed across ALL GPUs on ALL nodes.
#     Only the master node (rank 0) combines final results.
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

echo "=== Multi-Node Evaluation Configuration (Rubric Scoring) ==="
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "WORLD_SIZE (total GPUs): $WORLD_SIZE"
echo "LOCAL_NUM_GPUS: $LOCAL_NUM_GPUS"
echo ""

### Parse arguments: values to change
## MODEL_PATH: Path to the model (base model for baseline, fine-tuned for evaluation)
# MODEL_PATH="${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-7B"  # Base model
MODEL_PATH="${HC_CHECKPOINT_DIR}"  # Fine-tuned HC model
## LORA_PATH
LORA_PATH=""
## OUTPUT_DIR
OUTPUT_DIR="${EVAL_OUTPUT_DIR}/hypothesis_composition"

## API max tokens for rubric evaluation output
API_MAX_TOKENS=32768

## Generation parameters (defaults match training distribution)
MAX_NEW_TOKENS=8192  # training max ~2900, so 4096 gives 40% buffer
TEMPERATURE=0.6      # for diversity
TOP_P=0.9
REPETITION_PENALTY=1.2  # to prevent repetition

## Batch size for better GPU utilization
## 7B model: 32-48 recommended
## 32B model: 8-16 recommended
BATCH_SIZE=48

## Parallel LLM evaluation workers (for scoring API calls)
## Should be >= batch_size to avoid eval becoming bottleneck
EVAL_MAX_WORKERS=96


# Parse arguments: Default values
SFT_QA_DATA_DIR="${TOMATO_STAR_EVAL_DIR}"
MODEL_NAME="R1-Distill-Qwen-32B"
API_TYPE=0
API_KEY="YOUR_API_KEY_HERE"
BASE_URL="${API_BASE_URL}"
OTHER_ARGS=()

echo "MODEL_PATH: $MODEL_PATH"
echo "LORA_PATH: $LORA_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "TEMPERATURE: $TEMPERATURE"
echo "REPETITION_PENALTY: $REPETITION_PENALTY"
echo "API_MAX_TOKENS: $API_MAX_TOKENS"
echo "EVAL_MAX_WORKERS: $EVAL_MAX_WORKERS"

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
        --sft_qa_data_dir)
            SFT_QA_DATA_DIR="$2"
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
if [ -z "$MODEL_PATH" ] || [ -z "$SFT_QA_DATA_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 --model_path PATH --sft_qa_data_dir PATH --eval_result_path PATH [--lora_path PATH] [other args]"
    exit 1
fi

# Create output directory (shared across all nodes)
mkdir -p "$(dirname "$OUTPUT_DIR")"
TEMP_DIR="${OUTPUT_DIR%.json}_temp"
mkdir -p "$TEMP_DIR"

# Get list of JSON files from sft_qa_data_dir
eval_files=($(ls "$SFT_QA_DATA_DIR"/*.json 2>/dev/null | xargs -n1 basename | sort))
num_files=${#eval_files[@]}

if [ $num_files -eq 0 ]; then
    echo "No JSON files found in $SFT_QA_DATA_DIR"
    exit 1
fi

# Save actual world size before unsetting for torch distributed
ACTUAL_WORLD_SIZE=$WORLD_SIZE

echo "Found $num_files files to process across $ACTUAL_WORLD_SIZE GPUs"

# If only 1 GPU total and 1 node, run sequentially
if [ "$ACTUAL_WORLD_SIZE" -eq 1 ] || [ $num_files -eq 1 ]; then
    echo "Running sequentially (1 GPU or 1 file)"
    # Note: env -u WORLD_SIZE prevents transformers from auto-enabling tensor parallelism
    env -u WORLD_SIZE python ${PROJECT_ROOT}/Evaluation/hypothesis_composition_eval_rubric.py \
        --model_path "$MODEL_PATH" \
        ${LORA_PATH:+--lora_path "$LORA_PATH"} \
        --sft_qa_data_dir "$SFT_QA_DATA_DIR" \
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
# Files are distributed round-robin across ALL global GPUs
# Global GPU ID = NODE_RANK * GPUS_PER_NODE + LOCAL_GPU_ID

echo "Node $NODE_RANK: Distributing files across $LOCAL_NUM_GPUS local GPUs"

# Start parallel processes on this node
pids=()
for local_gpu_id in $(seq 0 $((LOCAL_NUM_GPUS - 1))); do
    # Calculate global GPU ID
    global_gpu_id=$((NODE_RANK * GPUS_PER_NODE + local_gpu_id))
    
    # Collect files for this global GPU (round-robin across all GPUs in cluster)
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
    
    # Create a temporary directory with only this GPU's files
    gpu_qa_dir="${TEMP_DIR}/qa_node${NODE_RANK}_gpu${local_gpu_id}"
    mkdir -p "$gpu_qa_dir"
    
    # Copy files to GPU-specific directories
    for file in "${gpu_files[@]}"; do
        cp "$SFT_QA_DATA_DIR/$file" "$gpu_qa_dir/" 2>/dev/null || true
    done
    
    # Run evaluation on this GPU (output is now a folder)
    gpu_output="${TEMP_DIR}/result_node${NODE_RANK}_gpu${local_gpu_id}"
    echo "  GPU $local_gpu_id (global: $global_gpu_id): Processing ${#gpu_files[@]} files"
    
    # Note: env -u WORLD_SIZE prevents transformers from auto-enabling tensor parallelism
    CUDA_VISIBLE_DEVICES=$local_gpu_id env -u WORLD_SIZE python ${PROJECT_ROOT}/Evaluation/hypothesis_composition_eval_rubric.py \
        --model_path "$MODEL_PATH" \
        ${LORA_PATH:+--lora_path "$LORA_PATH"} \
        --sft_qa_data_dir "$gpu_qa_dir" \
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
    
    # Combine results from all nodes (RUBRIC SCORING VERSION)
    echo "Combining results from all nodes..."
    python -c "
import json
import glob
import os

temp_dir = '$TEMP_DIR'
output_folder = '$OUTPUT_DIR'.rstrip('.json')  # Remove .json if present
os.makedirs(output_folder, exist_ok=True)

# Find all GPU result folders from all nodes
gpu_folders = glob.glob(os.path.join(temp_dir, 'result_node*_gpu*'))
gpu_folders = [f for f in gpu_folders if os.path.isdir(f)]

if not gpu_folders:
    print(f'WARNING: No GPU result folders found in {temp_dir}')
    print('Check GPU logs for errors.')

all_metrics = []
all_generations = []
combined_summary = {
    'mean_total_score': None,
    'mean_motivation_score': None,
    'mean_mechanism_score': None,
    'mean_methodology_score': None,
    'min_total_score': None,
    'max_total_score': None,
    'average_hypothesis_length': None,
    'total_evaluations': 0,
    'total_samples_processed': 0,
    'extraction_failures': 0,
    'total_evaluations_attempted': 0,
    'num_nodes': $NNODES,
    'total_gpus': $ACTUAL_WORLD_SIZE
}

# Collect data from each GPU folder
for gpu_folder in sorted(gpu_folders):
    # Read metrics
    metrics_path = os.path.join(gpu_folder, 'metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_metrics.extend(data)
        except Exception as e:
            print(f'Error reading {metrics_path}: {e}')
    
    # Read generations
    gen_path = os.path.join(gpu_folder, 'generations.json')
    if os.path.exists(gen_path):
        try:
            with open(gen_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_generations.extend(data)
        except Exception as e:
            print(f'Error reading {gen_path}: {e}')
    
    # Read summary and aggregate
    summary_path = os.path.join(gpu_folder, 'summary.json')
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
                combined_summary['total_samples_processed'] += summary.get('total_samples_processed', 0)
                combined_summary['extraction_failures'] += summary.get('extraction_failures', 0)
                combined_summary['total_evaluations_attempted'] += summary.get('total_evaluations_attempted', 0)
                # Track min/max scores
                if summary.get('min_total_score') is not None:
                    if combined_summary['min_total_score'] is None or summary['min_total_score'] < combined_summary['min_total_score']:
                        combined_summary['min_total_score'] = summary['min_total_score']
                if summary.get('max_total_score') is not None:
                    if combined_summary['max_total_score'] is None or summary['max_total_score'] > combined_summary['max_total_score']:
                        combined_summary['max_total_score'] = summary['max_total_score']
        except Exception as e:
            print(f'Error reading {summary_path}: {e}')

# Save combined metrics
with open(os.path.join(output_folder, 'metrics.json'), 'w') as f:
    json.dump(all_metrics, f, indent=2)

# Save combined generations
with open(os.path.join(output_folder, 'generations.json'), 'w') as f:
    json.dump(all_generations, f, indent=2)

# Calculate combined summary statistics (RUBRIC SCORING VERSION)
# If scores dict exists, it MUST contain all three keys - direct access to catch bugs
all_total_scores = [m['total_score'] for m in all_metrics if m.get('total_score') is not None]
all_motivation_scores = [m['scores']['motivation'] for m in all_metrics if m.get('scores')]
all_mechanism_scores = [m['scores']['mechanism'] for m in all_metrics if m.get('scores')]
all_methodology_scores = [m['scores']['methodology'] for m in all_metrics if m.get('scores')]

# Get hypothesis word counts from generations.json
all_hyp_lengths = [len(g['generated_hypothesis'].split()) for g in all_generations 
                   if g.get('generated_hypothesis') and not g.get('extraction_failed', False)]

# Update total_evaluations to match successful metrics count
combined_summary['total_evaluations'] = len(all_total_scores)

if all_total_scores:
    combined_summary['mean_total_score'] = sum(all_total_scores) / len(all_total_scores)
if all_motivation_scores:
    combined_summary['mean_motivation_score'] = sum(all_motivation_scores) / len(all_motivation_scores)
if all_mechanism_scores:
    combined_summary['mean_mechanism_score'] = sum(all_mechanism_scores) / len(all_mechanism_scores)
if all_methodology_scores:
    combined_summary['mean_methodology_score'] = sum(all_methodology_scores) / len(all_methodology_scores)
if all_hyp_lengths:
    combined_summary['average_hypothesis_length'] = sum(all_hyp_lengths) / len(all_hyp_lengths)

# Save combined summary
with open(os.path.join(output_folder, 'summary.json'), 'w') as f:
    json.dump(combined_summary, f, indent=2)

print(f'Combined {len(all_metrics)} results from {len(gpu_folders)} GPU processes into {output_folder}')
print(f'  - metrics.json: {len(all_metrics)} entries')
print(f'  - generations.json: {len(all_generations)} entries')
mean_score = combined_summary['mean_total_score']
mean_score_str = f'{mean_score:.2f}/12' if mean_score is not None else 'N/A'
print(f'  - summary.json: mean_total_score={mean_score_str}')
motivation = combined_summary['mean_motivation_score']
mechanism = combined_summary['mean_mechanism_score']
methodology = combined_summary['mean_methodology_score']
mot_str = f'{motivation:.2f}' if motivation is not None else 'N/A'
mec_str = f'{mechanism:.2f}' if mechanism is not None else 'N/A'
met_str = f'{methodology:.2f}' if methodology is not None else 'N/A'
print(f'    Dimension scores: Motivation={mot_str}, Mechanism={mec_str}, Methodology={met_str}')
print(f'  - Nodes: {combined_summary[\"num_nodes\"]}, Total GPUs: {combined_summary[\"total_gpus\"]}')
"

    # Cleanup temp directory (only on master)
    rm -rf "$TEMP_DIR"
    
    echo ""
    echo "=== Evaluation complete (Rubric Scoring). Results saved to $OUTPUT_DIR ==="
else
    echo "Node $NODE_RANK: Finished. Master node will combine results."
fi

