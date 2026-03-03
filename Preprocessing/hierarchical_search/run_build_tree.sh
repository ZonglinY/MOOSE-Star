#!/bin/bash
# =============================================================================
# Build Hierarchical Search Tree
# 
# This script builds a hierarchical tree from scientific papers for efficient
# inspiration retrieval with O(log N) complexity.
#
# Input: SFT QA directory containing JSON files with inspirations
#
# Output:
#   - hierarchical_tree.json: tree structure
#   - embeddings.npy: SPECTER2 embeddings (cached)
#   - papers.json: paper metadata (id, title, abstract, year)
#   - tree_config.json: configuration and statistics
#   - papers_hash.txt: hash for cache validation
# =============================================================================

set -e

# Activate environment (set PROJECT_ROOT before running)
cd ${PROJECT_ROOT}

# =============================================================================
# Configuration
# =============================================================================

# Input: SFT QA data directory (contains JSON files with inspirations);
#   Build Inspiration Corpus by combining all inspirations in the test set.
SFT_QA_DIR="${TOMATO_STAR_EVAL_DIR}"

# Output directory
OUTPUT_DIR="${SEARCH_TREE_DIR}"

# Tree building parameters
BRANCHING_FACTOR=15    # Max children per node (match IR model's 15-select-1)

# Embedding parameters
BATCH_SIZE=64          # Batch size for SPECTER2 embedding
DEVICE="cuda"          # Device for embedding model (cuda/cpu)

# Other
SEED=42
USE_MEDOID=1           # 1: medoid selection (recommended), 0: closest to mean

# =============================================================================
# Run
# =============================================================================

echo "=============================================="
echo "Building Hierarchical Search Tree"
echo "=============================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Branching factor: ${BRANCHING_FACTOR}"
echo "=============================================="

mkdir -p "${OUTPUT_DIR}"

echo "Source: ${SFT_QA_DIR}"
python ${PROJECT_ROOT}/Preprocessing/hierarchical_search/build_hierarchical_tree.py \
    --sft_qa_dir "${SFT_QA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --branching_factor ${BRANCHING_FACTOR} \
    --use_medoid ${USE_MEDOID} \
    --batch_size ${BATCH_SIZE} \
    --device "${DEVICE}" \
    --seed ${SEED}

echo ""
echo "=============================================="
echo "Done! Output files:"
echo "  - ${OUTPUT_DIR}/hierarchical_tree.json"
echo "  - ${OUTPUT_DIR}/embeddings.npy"
echo "  - ${OUTPUT_DIR}/papers.json"
echo "  - ${OUTPUT_DIR}/tree_config.json"
echo "=============================================="

