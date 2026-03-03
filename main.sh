#!/bin/bash
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  MOOSE-Star: Training, Evaluation, and Inference Pipeline                ║
# ║                                                                          ║
# ║  Steps 3-5: SFT Training → Evaluation → Hierarchical Search Inference   ║
# ║                                                                          ║
# ║  Prerequisites:                                                          ║
# ║  - Download SFT training data from HuggingFace:                          ║
# ║      huggingface-cli download ZonglinY/TOMATO-Star-SFT-Data-R1D-32B \    ║
# ║          --local-dir ./data/sft_data                                     ║
# ║    This creates ./data/sft_data/HC/ and ./data/sft_data/IR/              ║
# ║                                                                          ║
# ║  - Download TOMATO-Star eval data:                                       ║
# ║      huggingface-cli download ZonglinY/TOMATO-Star \                     ║
# ║          --local-dir ./data/tomato_star                                  ║
# ║    Then convert test.jsonl → individual JSON files (see Step 0 below)    ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#
# ┌────────────────────────────────────────────────────────────────────────────┐
# │  Legend                                                                    │
# │                                                                            │
# │  【Input】 : Input data/files required by this step                        │
# │  【Output】: Output data/files produced, and downstream consumers          │
# │                                                                            │
# │  "-> used by Step X" means the output feeds into Step X                   │
# └────────────────────────────────────────────────────────────────────────────┘


# ============================================================
# Configuration — Update these paths for your environment
# ============================================================

# Base directories
export MODEL_DIR="/path/to/models"                    # HuggingFace model cache / base model dir
export DATA_DIR="/path/to/data"                       # General data directory

# TOMATO-Star eval data (download from HuggingFace: ZonglinY/TOMATO-Star)
# After download, run the conversion below to split test.jsonl into individual JSON files per paper.
export TOMATO_STAR_EVAL_DIR="${DATA_DIR}/tomato_star/eval"  # directory of individual .json files (one per paper)

# SFT training data directories (download from HuggingFace: ZonglinY/TOMATO-Star-SFT-Data-R1D-32B)
# HC SFT data: the HC/ subdirectory of the downloaded SFT data
#   Contains: dataset_info.json, normal_composition.jsonl, bounded_composition.jsonl
export HC_MIXED_SFT_DIR="${DATA_DIR}/sft_data/HC"
# IR SFT data: the IR/ subdirectory of the downloaded SFT data
#   Contains: dataset_info.json, train.jsonl, eval.jsonl
export IR_SFT_DATA_DIR="${DATA_DIR}/sft_data/IR"

# Bounded inspiration selections for Step 4.2 (not publicly released; skip Step 4.2 if unavailable)
export BOUNDED_SELECTIONS_DIR="${DATA_DIR}/bounded_selections"

# Base model (DeepSeek-R1-Distill-Qwen-7B)
export BASE_MODEL="${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-7B"

# Output directories
export HC_CHECKPOINT_DIR="${DATA_DIR}/checkpoints/hypothesis_composition"
export IR_CHECKPOINT_DIR="${DATA_DIR}/checkpoints/inspiration_retrieval"
export EVAL_OUTPUT_DIR="${DATA_DIR}/evaluation_results"
export SEARCH_TREE_DIR="${DATA_DIR}/hierarchical_search_tree"
export SEARCH_EVAL_DIR="${DATA_DIR}/search_eval_results"

# API settings (for LLM-based rubric scoring in Step 4; any OpenAI-compatible endpoint)
export API_BASE_URL="http://localhost:8000/v1"        # Scoring LLM endpoint
export API_KEY="YOUR_API_KEY_HERE"                    # API key for scoring LLM


# ══════════════════════════════════════════════════════════════════════════════
#  0. DATA PREPARATION (One-time Setup)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Convert TOMATO-Star test.jsonl → individual JSON files (one per paper).
#  Evaluation scripts (Steps 4.1/4.2 and Step 5) read ${TOMATO_STAR_EVAL_DIR}
#  and expect a directory of individual .json files named by source_id.
#
#  Run this once after downloading TOMATO-Star from HuggingFace.
#
# 【Input】: ./data/tomato_star/test.jsonl (downloaded from ZonglinY/TOMATO-Star)
# 【Output】: ${TOMATO_STAR_EVAL_DIR}/*.json (one file per paper)
# ------------------------------------------------------------------------------
mkdir -p "${TOMATO_STAR_EVAL_DIR}"
python3 - <<'EOF'
import json, os, sys

src = os.path.join(os.environ["DATA_DIR"], "tomato_star", "test.jsonl")
dst = os.environ["TOMATO_STAR_EVAL_DIR"]

if not os.path.exists(src):
    print(f"ERROR: {src} not found. Download TOMATO-Star first:")
    print("  huggingface-cli download ZonglinY/TOMATO-Star --local-dir ./data/tomato_star")
    sys.exit(1)

count = 0
with open(src, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        paper = json.loads(line)
        paper_id = paper.get("source_id", f"paper_{count}")
        out_path = os.path.join(dst, f"{paper_id}.json")
        with open(out_path, "w") as out:
            json.dump(paper, out, ensure_ascii=False, indent=2)
        count += 1

print(f"Converted {count} papers from test.jsonl → {dst}/")
EOF


# ══════════════════════════════════════════════════════════════════════════════
#  3. SFT TRAINING (Supervised Fine-Tuning)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Supervised fine-tuning using LLaMA-Factory.
#  Full-parameter fine-tuning with multi-node distributed training.
#
#  【Data Flow】
#    Input:  TOMATO-Star-SFT-Data-R1D-32B (HC portion + IR portion)
#    Output: Model Checkpoint -> used by Step 4 evaluation and Step 5 inference
#
# ══════════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------------------
# 3.1 Hypothesis Composition Training (Full Parameter)
# ------------------------------------------------------------------------------
# Functionality:
#   - Multi-node distributed full-parameter training
#   - Uses DeepSpeed ZeRO-3 for memory optimization
#
# Parameters (in SFT/full_train_hypothesis_composition.yaml):
#   - model_name_or_path: Base model path (${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-7B)
#   - dataset_dir: HC SFT data directory (${HC_MIXED_SFT_DIR})
#   - output_dir: HC checkpoint output (${HC_CHECKPOINT_DIR})
#   - deepspeed: DeepSpeed config file
#   - per_device_train_batch_size: Batch size per GPU
#
# Environment variables (set by your cluster scheduler, or set manually):
#   - MLP_ROLE_INDEX (or NODE_RANK): Node rank (0-indexed)
#   - MLP_WORKER_0_HOST (or MASTER_ADDR): Master node IP
#   - MLP_WORKER_NUM (or NNODES): Number of nodes
#   - MLP_WORKER_GPU (or NPROC_PER_NODE): GPUs per node
#
# Training configuration:
#   - Base model: DeepSeek-R1-Distill-Qwen-7B
#   - Template: deepseekr1
#   - Effective batch size: 128 (1 per device × 2 gradient accumulation × 64 GPUs)
#   - Learning rate: 1e-5 with cosine schedule
#   - 1 epoch, DeepSpeed ZeRO-3, bf16
#
# Note: ${HC_MIXED_SFT_DIR} should point to the HC/ subdirectory from TOMATO-Star-SFT-Data-R1D-32B.
#       It contains dataset_info.json (keys: "normal_composition", "bounded_composition"),
#       normal_composition.jsonl, and bounded_composition.jsonl.
#
# 【Input】: HC SFT Data (${HC_MIXED_SFT_DIR}/normal_composition.jsonl + bounded_composition.jsonl)
# 【Output】: HC model Checkpoint (${HC_CHECKPOINT_DIR}) -> used by Step 4.1/4.2 evaluation
# ------------------------------------------------------------------------------
bash SFT/train_hypothesis_composition.sh


# ------------------------------------------------------------------------------
# 3.2 Inspiration Retrieval Training (Full Parameter)
# ------------------------------------------------------------------------------
# Functionality:
#   - Multi-node distributed full-parameter training
#   - Train model to select the correct inspiration from 15 candidates
#   - Uses DeepSpeed ZeRO-3 for memory optimization
#
# Parameters (in SFT/full_train_inspiration_retrieval.yaml):
#   - model_name_or_path: Base model path (${MODEL_DIR}/DeepSeek-R1-Distill-Qwen-7B)
#   - dataset_dir: IR SFT data directory (${IR_SFT_DATA_DIR})
#   - output_dir: IR checkpoint output (${IR_CHECKPOINT_DIR})
#   - deepspeed: DeepSpeed config file
#   - per_device_train_batch_size: Batch size per GPU
#
# Environment variables: Same as 3.1
#
# Training configuration:
#   - Base model: DeepSeek-R1-Distill-Qwen-7B
#   - Template: deepseekr1
#   - Effective batch size: 128 (1 per device × 1 gradient accumulation × 128 GPUs)
#   - Sequence length: 16384
#   - 1 epoch, DeepSpeed ZeRO-3, bf16
#
# Note: Ensure ${IR_SFT_DATA_DIR} contains dataset_info.json (maps "train"/"eval" to actual JSONL files).
#       This file is included in the downloaded TOMATO-Star-SFT-Data-R1D-32B.
#
# 【Input】: IR SFT Data (${IR_SFT_DATA_DIR}/train.jsonl)
# 【Output】: IR model Checkpoint (${IR_CHECKPOINT_DIR}) -> used by Step 4.3 and Step 5
# ------------------------------------------------------------------------------
bash SFT/train_inspiration_retrieval.sh


# ══════════════════════════════════════════════════════════════════════════════
#  4. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
#
#  Evaluate trained models on two core tasks:
#    - Hypothesis Composition: Evaluate hypothesis generation quality
#    - Inspiration Retrieval: Evaluate inspiration retrieval accuracy
#
#  Supports multi-node parallel evaluation with Rubric scoring.
#
#  【Data Flow】
#    Input:
#      - HC evaluation: Step 3.1 HC model + TOMATO-Star eval data
#      - IR evaluation: Step 3.2 IR model + IR eval data (eval.jsonl from TOMATO-Star-SFT-Data-R1D-32B)
#    Output: Evaluation result JSON (metrics.json, summary.json)
#
# ══════════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------------------
# 4.1 Hypothesis Composition Evaluation (Normal Mode)
# ------------------------------------------------------------------------------
# Functionality:
#   - Evaluate model hypothesis generation quality using rubric scoring
#   - Rubric scoring via an LLM API (Motivation, Mechanism, Methodology each 0-4 points)
#   - Supports multi-node parallel evaluation
#
# Configure MODEL_PATH, OUTPUT_DIR, and API settings in the script:
#   Evaluation/run_hypothesis_composition_eval_rubric_parallel.sh
#
# Key parameters (set at top of shell script):
#   - MODEL_PATH: HC model checkpoint path (use ${HC_CHECKPOINT_DIR})
#   - OUTPUT_DIR: Evaluation results directory (use ${EVAL_OUTPUT_DIR}/hc_normal)
#   - SFT_QA_DATA_DIR: TOMATO-Star eval data (use ${TOMATO_STAR_EVAL_DIR})
#   - API_KEY, API_BASE_URL: Scoring LLM API settings
#   - BATCH_SIZE: Batch size (7B model: 32-48 recommended)
#   - MAX_NEW_TOKENS: Maximum generation length
#
# 【Input】: Step 3.1 HC model + TOMATO-Star eval data (${TOMATO_STAR_EVAL_DIR})
# 【Output】: metrics.json, generations.json, summary.json
# ------------------------------------------------------------------------------
bash Evaluation/run_hypothesis_composition_eval_rubric_parallel.sh


# ------------------------------------------------------------------------------
# 4.2 Hypothesis Composition Evaluation (Bounded Mode)
# ------------------------------------------------------------------------------
# Functionality:
#   - Evaluate model robustness to imperfect retrieval using bounded inspirations
#   - Bounded inspirations are semantically similar but not exact matches
#   - Reports statistics by difficulty tier:
#     * hard: similarity [0.90, 0.92) - largest difference from ground truth
#     * medium: similarity [0.92, 0.94)
#     * easy: similarity [0.94, 0.97) - most similar to ground truth
#
# Configure MODEL_PATH, BOUNDED_SELECTIONS_DIR, and OUTPUT_DIR in the script:
#   Evaluation/run_hypothesis_composition_eval_rubric_bounded_parallel.sh
#
# Key parameters (set at top of shell script):
#   - MODEL_PATH: HC model checkpoint path (use ${HC_CHECKPOINT_DIR})
#   - BOUNDED_SELECTIONS_DIR: Bounded inspirations (use ${BOUNDED_SELECTIONS_DIR})
#   - OUTPUT_DIR: Evaluation results directory (use ${EVAL_OUTPUT_DIR}/hc_bounded)
#
# 【Input】: Step 3.1 HC model + TOMATO-Star eval data + Bounded Selections
# 【Output】: Per-tier metrics/generations/summary JSON
# ------------------------------------------------------------------------------
bash Evaluation/run_hypothesis_composition_eval_rubric_bounded_parallel.sh


# ------------------------------------------------------------------------------
# 4.3 Inspiration Retrieval Evaluation
# ------------------------------------------------------------------------------
# Functionality:
#   - Evaluate model accuracy in selecting the correct inspiration from 15 candidates
#   - Uses the IR eval data from TOMATO-Star-SFT-Data-R1D-32B (eval.jsonl)
#   - Supports multiple model modes (base, LoRA, full fine-tuned)
#
# Configure MODEL_PATH, DATA_FILE, and OUTPUT_DIR in the script:
#   Evaluation/run_inspiration_retrieval_eval_parallel.sh
#
# Key parameters (set at top of shell script):
#   - MODEL_PATH: IR model checkpoint path (use ${IR_CHECKPOINT_DIR})
#   - DATA_FILE: IR eval data (use ${IR_SFT_DATA_DIR}/eval.jsonl)
#   - OUTPUT_DIR: Evaluation results directory (use ${EVAL_OUTPUT_DIR}/ir)
#   - MAX_RETRIES: Extraction failure retry count (recommend 2-3)
#
# 【Input】: Step 3.2 IR model + IR eval data (${IR_SFT_DATA_DIR}/eval.jsonl)
# 【Output】: eval_results_combined.json (accuracy, per-sample predictions)
# ------------------------------------------------------------------------------
nohup bash Evaluation/run_inspiration_retrieval_eval_parallel.sh > ./Logs/inspiration_retrieval_eval.log 2>&1 &


# ══════════════════════════════════════════════════════════════════════════════
#  5. HIERARCHICAL SEARCH INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
#
#  Use the trained IR model to perform hierarchical search over an inspiration corpus.
#
#  Search algorithms:
#    - Hierarchical Search (Best-First): O(log N) inference calls, efficient
#    - Tournament Search (Baseline): O(N/K) inference calls, visits all leaf nodes
#
#  Requires:
#    1. Building the SPECTER2-clustered search tree (one-time)
#    2. Deploying an SGLang inference service
#
#  【Data Flow】
#    Input:  Step 3.2 IR model + Inspiration corpus (built into Search Tree)
#    Output: Search result JSON (top-K retrieved inspirations, accuracy metrics)
#
# ══════════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------------------
# 5.0 Build the Hierarchical Search Tree (one-time setup)
# ------------------------------------------------------------------------------
# Functionality:
#   - Build a SPECTER2-clustered hierarchical tree from the inspiration corpus
#   - The corpus is derived from TOMATO-Star eval data (all unique inspirations)
#   - Tree structure enables O(log N) search vs O(N) brute force
#
# Parameters (set in Preprocessing/hierarchical_search/run_build_tree.sh):
#   - SFT_QA_DIR: TOMATO-Star eval directory (use ${TOMATO_STAR_EVAL_DIR})
#   - OUTPUT_DIR: Tree output directory (use ${SEARCH_TREE_DIR})
#   - BRANCHING_FACTOR: Max children per node (default 15, matches IR model's 15-way selection)
#
# Output files:
#   - hierarchical_tree.json: Tree structure
#   - embeddings.npy: SPECTER2 embeddings (cached)
#   - papers.json: Paper metadata (id, title, abstract, year)
#   - tree_config.json: Configuration and statistics
#
# 【Input】: TOMATO-Star eval data (${TOMATO_STAR_EVAL_DIR})
# 【Output】: Search tree files (${SEARCH_TREE_DIR}) -> used by Steps 5.2/5.3
# ------------------------------------------------------------------------------
export PROJECT_ROOT=$(pwd)
bash Preprocessing/hierarchical_search/run_build_tree.sh


# ------------------------------------------------------------------------------
# 5.1 Deploy SGLang Inference Service
# ------------------------------------------------------------------------------
# Functionality:
#   - Deploy IR model as an OpenAI-compatible inference server
#   - 7B model uses TP=2 (2 GPUs per instance)
#   - Service listens on the specified port
#
# Note: Install SGLang separately before this step:
#   pip install "sglang[all]>=0.5.0"
#
# Usage:
#   bash Inference/start_sglang.sh              # Start on default port 1235
#   bash Inference/start_sglang.sh --port 8000  # Custom port
#
# Multi-node scaling (for higher throughput):
#   1. SSH to each node and run start_sglang.sh with different ports
#   2. Combine all node URLs in --sglang-urls parameter (Steps 5.2/5.3)
#
# 【Input】: Step 3.2 IR model Checkpoint (${IR_CHECKPOINT_DIR})
# 【Output】: SGLang inference service (HTTP API) -> used by Steps 5.2/5.3
# ------------------------------------------------------------------------------
bash Inference/start_sglang.sh


# ------------------------------------------------------------------------------
# 5.2 Hierarchical Search Evaluation (Best-First Search)
# ------------------------------------------------------------------------------
# Functionality:
#   - Use trained IR model to perform hierarchical best-first search
#   - Uses geometric mean probability to compare nodes at different depths
#   - Evaluates whether the correct inspiration can be found
#   - Complexity: O(log N) inference calls (~3 calls for N=3035, K=15)
#
# Parameters:
#   --tree-dir: Hierarchical search tree directory
#   --eval-dir: TOMATO-Star eval data directory
#   --sglang-urls: SGLang service endpoints (can be multiple for load balancing)
#   --search-mode: Search mode (best_first recommended)
#   --max-samples: Number of evaluation samples
#   --max-proposals: Maximum proposals (0=unlimited, search until found)
#   --num-workers: Concurrent worker count (recommend >= max-samples)
#   --softmax-temperature: Probability scaling temperature
#   --max-tokens: LLM output max length
#   --top-logprobs: Number of top logprobs to return
#   --truncate-survey: Survey truncation option
#     * 0: Full survey (default)
#     * 1: Truncated survey (remove "However, limitation..." implicit motivation)
#   --motivation-option: Motivation prompt option
#     * 0: None (baseline)
#     * 1: Simple (Problem/Gap + Solution Direction)
#     * 2: Detailed (full Motivation WHY)
#   --output-dir: Output directory
#   --resume: Support resumption
#
# Ablation experiment matrix (6 combinations):
#   TRUNCATE=0 + MOTIVATION=0 -> baseline
#   TRUNCATE=0 + MOTIVATION=1 -> simple_motivation
#   TRUNCATE=0 + MOTIVATION=2 -> detailed_motivation
#   TRUNCATE=1 + MOTIVATION=0 -> truncated_baseline
#   TRUNCATE=1 + MOTIVATION=1 -> truncated_simple_motivation
#   TRUNCATE=1 + MOTIVATION=2 -> truncated_detailed_motivation
#
# 【Input】: Step 5.1 SGLang service + Search Tree + TOMATO-Star eval data
# 【Output】: Search result JSON (top-K retrieved, accuracy@K)
# ------------------------------------------------------------------------------
# Experiment configuration (modify these two variables)
TRUNCATE_SURVEY=1    # 0=full survey, 1=truncated survey (remove implicit motivation)
MOTIVATION_OPTION=2  # 0=baseline, 1=simple motivation, 2=detailed motivation

# Auto-generate output suffix
if [ "$TRUNCATE_SURVEY" -eq 1 ]; then
    TRUNCATE_PREFIX="truncated_survey_"
else
    TRUNCATE_PREFIX="full_survey_"
fi

if [ "$MOTIVATION_OPTION" -eq 0 ]; then
    MOTIVATION_SUFFIX="baseline"
elif [ "$MOTIVATION_OPTION" -eq 1 ]; then
    MOTIVATION_SUFFIX="simple_motivation"
else
    MOTIVATION_SUFFIX="detailed_motivation"
fi

OUTPUT_SUFFIX="${TRUNCATE_PREFIX}${MOTIVATION_SUFFIX}"

nohup python -u Inference/hierarchical_search_eval.py \
    --tree-dir "${SEARCH_TREE_DIR}" \
    --eval-dir "${TOMATO_STAR_EVAL_DIR}" \
    --sglang-urls "${API_BASE_URL}" \
    --search-mode best_first \
    --max-samples 200 \
    --max-proposals 0 \
    --num-workers 200 \
    --softmax-temperature 5.0 \
    --max-tokens 20480 \
    --top-logprobs 30 \
    --truncate-survey $TRUNCATE_SURVEY \
    --motivation-option $MOTIVATION_OPTION \
    --output-dir "${SEARCH_EVAL_DIR}/hierarchical_search_eval_maxsample_200_${OUTPUT_SUFFIX}" \
    --resume \
    > ./Logs/hierarchical_search_eval_maxsample_200_${OUTPUT_SUFFIX}.log 2>&1 &


# ------------------------------------------------------------------------------
# 5.3 Tournament Search Evaluation (Baseline)
# ------------------------------------------------------------------------------
# Functionality:
#   - Bottom-up tournament-style search, visiting all leaf nodes
#   - Serves as O(N/K) baseline comparison for hierarchical search
#   - Complexity: O(N/K) inference calls (~218 calls for N=3035, K=15)
#   - Hierarchical search complexity for comparison: O(log N) (~3 calls)
#
# Parameters: Same as 5.2
#
# 【Input】: Step 5.1 SGLang service + Search Tree + TOMATO-Star eval data
# 【Output】: Tournament search result JSON (compare with 5.2)
# ------------------------------------------------------------------------------
# Experiment configuration (keep consistent with 5.2 for fair comparison)
TOURNAMENT_TRUNCATE_SURVEY=1    # 0=full survey, 1=truncated survey
TOURNAMENT_MOTIVATION_OPTION=0  # 0=baseline, 1=simple motivation, 2=detailed motivation

# Auto-generate output suffix
if [ "$TOURNAMENT_TRUNCATE_SURVEY" -eq 1 ]; then
    TOURNAMENT_TRUNCATE_PREFIX="truncated_survey_"
else
    TOURNAMENT_TRUNCATE_PREFIX="full_survey_"
fi

if [ "$TOURNAMENT_MOTIVATION_OPTION" -eq 0 ]; then
    TOURNAMENT_MOTIVATION_SUFFIX="baseline"
elif [ "$TOURNAMENT_MOTIVATION_OPTION" -eq 1 ]; then
    TOURNAMENT_MOTIVATION_SUFFIX="simple_motivation"
else
    TOURNAMENT_MOTIVATION_SUFFIX="detailed_motivation"
fi

TOURNAMENT_SUFFIX="${TOURNAMENT_TRUNCATE_PREFIX}${TOURNAMENT_MOTIVATION_SUFFIX}"

nohup python -u Inference/tournament_search_eval.py \
    --tree-dir "${SEARCH_TREE_DIR}" \
    --eval-dir "${TOMATO_STAR_EVAL_DIR}" \
    --sglang-urls "${API_BASE_URL}" \
    --max-samples 200 \
    --num-workers 200 \
    --softmax-temperature 5.0 \
    --max-tokens 20480 \
    --top-logprobs 30 \
    --truncate-survey $TOURNAMENT_TRUNCATE_SURVEY \
    --motivation-option $TOURNAMENT_MOTIVATION_OPTION \
    --output-dir "${SEARCH_EVAL_DIR}/tournament_search_maxsample_200_${TOURNAMENT_SUFFIX}" \
    --resume \
    > ./Logs/tournament_search_maxsample_200_${TOURNAMENT_SUFFIX}.log 2>&1 &
