#!/bin/bash
# Script for launching multi-node training for Hypothesis Composition

#############################################################################
# Multi-Node Training Script for Hypothesis Composition
# Supports both single-node and multi-node training
# 
# Usage:
#   Single-node (8 GPUs):
#     bash train_hypothesis_composition.sh
#     # Script auto-detects single-node when MLP_WORKER_NUM=1 or NNODES=1
#
#   Multi-node (64 GPUs = 8 nodes):
#     The script uses the following environment variables (commonly set by cluster schedulers):
#     - MLP_ROLE_INDEX: Node rank (0-7)
#     - MLP_WORKER_0_HOST: Master node IP address
#     - MLP_WORKER_0_PORT: Master port
#     - MLP_WORKER_NUM: Number of nodes (8)
#     - MLP_WORKER_GPU: GPUs per node (8)
#
#     If platform variables are not set, you can manually set:
#     - NODE_RANK: 0-7 for each node
#     - MASTER_ADDR: IP address of node 0
#     - NNODES: 8
#     - NPROC_PER_NODE: 8
#############################################################################

# Verify Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: python command not found after activating environment"
    echo "PATH: $PATH"
    echo "Trying to use python3..."
    if command -v python3 &> /dev/null; then
        alias python=python3
    else
        echo "ERROR: Neither python nor python3 found. Exiting."
        exit 1
    fi
fi
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# === MULTI-NODE CONFIGURATION ===
# Use platform environment variables with fallbacks (matching launch.sh approach)
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-29500}}
export NNODES=${MLP_WORKER_NUM:-${NNODES:-1}}
export NODE_RANK=${MLP_ROLE_INDEX:-${NODE_RANK:-0}}
export NPROC_PER_NODE=${MLP_WORKER_GPU:-${NPROC_PER_NODE:-8}}

# Compute total world size
export WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# Print configuration (matching launch.sh style)
echo "=== Multi-Node Configuration ==="
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MASTER_PORT: $MASTER_PORT"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "WORLD_SIZE: $WORLD_SIZE"
echo ""

# Print network interfaces for debugging
echo "=== Network Interfaces ==="
ip addr show 2>/dev/null | grep -E "^[0-9]+:|inet " | head -20 || ifconfig 2>/dev/null | head -20
echo ""

# Test connectivity to master node (from worker nodes)
if [ "$NODE_RANK" -ne 0 ]; then
    echo "=== Testing connectivity to master ($MASTER_ADDR:$MASTER_PORT) ==="
    timeout 5 bash -c "echo >/dev/tcp/$MASTER_ADDR/$MASTER_PORT" 2>/dev/null && echo "Master reachable!" || echo "Warning: Cannot reach master on port $MASTER_PORT (may be fine if master not yet started)"
    echo ""
fi

# Validate configuration
if [ "$NNODES" -lt 1 ]; then
    echo "ERROR: Number of nodes must be at least 1"
    exit 1
fi
if [ "$NODE_RANK" -ge "$NNODES" ]; then
    echo "ERROR: NODE_RANK ($NODE_RANK) must be less than NNODES ($NNODES)"
    echo "       Valid NODE_RANK values: 0 to $((NNODES - 1))"
    exit 1
fi

# GPU visibility
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL settings for multi-node training (matching working examples)
export NCCL_DEBUG=INFO  # More verbose for debugging (change to WARN after success)

# InfiniBand settings - USE InfiniBand if available
export NCCL_IB_TIMEOUT=80
export NCCL_IB_RETRY_CNT=20
export NCCL_IB_AR_THRESHOLD=0

# P2P settings
export NCCL_P2P_LEVEL=NVL  # Use NVLink if available

# Additional stability settings for multi-node
export NCCL_ASYNC_ERROR_HANDLING=1

# Diagnostic: Check InfiniBand availability
if [ "$NNODES" -gt 1 ]; then
    echo "=== InfiniBand Diagnostics ==="
    if command -v ibstat &> /dev/null; then
        echo "IB Status:"
        ibstat 2>/dev/null | head -20 || echo "  ibstat failed"
    else
        echo "  ibstat command not available"
    fi
    
    if [ -d /sys/class/infiniband ]; then
        echo "IB Devices:"
        for dev in /sys/class/infiniband/*/ports/*/state; do
            if [ -f "$dev" ]; then
                echo "  $dev: $(cat $dev)"
            fi
        done
    else
        echo "  No /sys/class/infiniband directory found"
    fi
    
    echo "NCCL Environment Variables:"
    env | grep -E "^NCCL_" | sort
    echo ""
fi

# if [ "$NNODES" -eq 1 ]; then
#     # Single-node specific overrides (disabled - causes issues with InfiniBand)
#     export NCCL_P2P_DISABLE=1
#     export NCCL_NET_PLUGIN=none
# fi

# PyTorch settings
export OMP_NUM_THREADS=1
# Memory management: expandable_segments helps with fragmentation on long sequences
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0  # Async CUDA operations (set to 1 for debugging)

# Prevent OOM-related SIGSEGV by reserving memory
export CUDA_MEMORY_POOL_LOGGING=1

cd ${PROJECT_ROOT:-.}
mkdir -p Logs

# Use torchrun for proper multi-node distributed training
# LLaMA-Factory integrates with torchrun via FORCE_TORCHRUN=1
export FORCE_TORCHRUN=1

# Generate log filename based on configuration
if [ "$NNODES" -eq 1 ]; then
    LOG_FILE="Logs/training_hypothesis_composition_${WORLD_SIZE}gpu.log"
else
    LOG_FILE="Logs/training_hypothesis_composition_${WORLD_SIZE}gpu_node${NODE_RANK}.log"
fi

# Use python -m torch.distributed.run instead of torchrun directly
# to avoid hardcoded shebang interpreter issues
# Use static rendezvous backend for more reliable multi-node coordination
echo "=== Starting Training ==="
echo "Log file: $LOG_FILE"
echo ""

# Use explicit Python path from activated environment
PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN=$(which python)
fi
echo "Executing with: $PYTHON_BIN"

$PYTHON_BIN -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m llamafactory.launcher \
    SFT/full_train_hypothesis_composition.yaml \
    2>&1 | tee $LOG_FILE

echo "Training completed on node $NODE_RANK"
