#!/bin/bash
# ==============================================================================
# Start SGLang server for IR model inference
# ==============================================================================
# Deploys the IR model as an OpenAI-compatible inference server.
# The 7B IR model fits comfortably on a single 40GB+ GPU (tp=1).
# Use tp=2 for faster throughput (requires 2 GPUs).
#
# Usage:
#   bash Inference/start_sglang.sh                    # tp=1, port 1235
#   bash Inference/start_sglang.sh --port 8000        # custom port
#   bash Inference/start_sglang.sh --tp 2             # 2-GPU tensor parallelism
#
# Config: context=32k, host=0.0.0.0 (accepts external connections)
# ==============================================================================

set -e

# Parse arguments
PORT=1235  # Default port
TP=1       # Default tensor parallelism (1 GPU is sufficient for 7B model)
MODEL_PATH="${IR_CHECKPOINT_DIR}"  # Default: from main.sh config
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --tp)
            TP="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Using Python: $(which python3)"

# CUDA paths for flashinfer JIT compilation
DISABLE_CUDA_GRAPH=""
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
elif [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME="/usr/local/cuda-12.8"
elif [ -d "/usr/local/cuda-12" ]; then
    export CUDA_HOME="/usr/local/cuda-12"
else
    echo "WARNING: CUDA toolkit not found. Will disable cuda-graph."
    DISABLE_CUDA_GRAPH="--disable-cuda-graph"
fi

if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    echo "CUDA_HOME: $CUDA_HOME"
fi

# Disable proxy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="*"
export no_proxy="*"

# Pre-flight checks
echo "=============================================="
echo "Starting SGLang Server (tp=${TP})"
echo "=============================================="
echo "Port: $PORT"
echo "Model: $MODEL_PATH"

if ss -tlnp 2>/dev/null | grep -q ":$PORT " || netstat -tlnp 2>/dev/null | grep -q ":$PORT "; then
    echo "ERROR: Port $PORT is already in use!"
    exit 1
fi
echo "✓ Port $PORT available"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    echo "  Set IR_CHECKPOINT_DIR or IR_MODEL_PATH to the IR model directory."
    exit 1
fi
echo "✓ Model path exists"

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "✓ Found $GPU_COUNT GPU(s), using tp=${TP}"
if [ "$GPU_COUNT" -lt "$TP" ]; then
    echo "WARNING: tp=${TP} requested but only $GPU_COUNT GPU(s) available"
fi

# Use chat_template.jinja if present in the model dir; otherwise let SGLang
# auto-detect the template from tokenizer_config.json (preferred for HF models)
CHAT_TEMPLATE_ARG=""
if [ -f "${MODEL_PATH}/chat_template.jinja" ]; then
    CHAT_TEMPLATE_ARG="--chat-template ${MODEL_PATH}/chat_template.jinja"
    echo "✓ Using chat_template.jinja from model directory"
else
    echo "✓ Using chat template from tokenizer_config.json (auto-detect)"
fi

echo ""
echo "Starting SGLang server..."
echo "=============================================="

python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp "$TP" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --context-length 32768 \
    $CHAT_TEMPLATE_ARG \
    --log-level warning \
    $DISABLE_CUDA_GRAPH
