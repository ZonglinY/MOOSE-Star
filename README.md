# MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Collection-ffd21e.svg)](https://huggingface.co/ZonglinY)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Abstract

MOOSE-Star is a framework for training LLMs to generate scientific hypotheses with tractable complexity. Prior methods require enumerating all possible inspiration combinations during training, leading to O(N^k) complexity. MOOSE-Star breaks this barrier by (1) decomposing hypothesis composition into sequential single-inspiration steps, (2) training a separate Inspiration Retrieval (IR) model, and (3) performing hierarchical search over a SPECTER2-clustered tree at inference time, reducing complexity to O(log N). Built on DeepSeek-R1-Distill-Qwen-7B with rejection sampling from a 32B teacher model, MOOSE-Star achieves 54.37% IR accuracy (vs. 28.42% base, 6.70% random) and a Hypothesis Composition total score of 5.16 (vs. 4.34 base).

## Released Resources

| Resource | Description | Link |
|---|---|---|
| TOMATO-Star | Decomposed biomedical papers dataset (107K train, 1.6K eval) | [HuggingFace](https://huggingface.co/datasets/ZonglinY/TOMATO-Star) |
| TOMATO-Star-SFT-Data-R1D-32B | SFT training data: HC train 96K + bounded train 17K + IR train 150K + IR eval 2.3K (HC eval uses TOMATO-Star directly) | [HuggingFace](https://huggingface.co/datasets/ZonglinY/TOMATO-Star-SFT-Data-R1D-32B) |
| MOOSE-Star-HC-R1D-7B | Hypothesis Composition model (MS-HC-7B in paper) | [HuggingFace](https://huggingface.co/ZonglinY/MOOSE-Star-HC-R1D-7B) |
| MOOSE-Star-IR-R1D-7B | Inspiration Retrieval model (MS-IR-7B in paper) | [HuggingFace](https://huggingface.co/ZonglinY/MOOSE-Star-IR-R1D-7B) |

## Installation

```bash
git clone https://github.com/ZonglinY/MOOSE-Star.git
cd MOOSE-Star
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.7+ and CUDA 12.8. Training requires [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (included in `requirements.txt`). Hierarchical search inference requires [SGLang](https://github.com/sgl-project/sglang) (installed separately; see Step 3).

## Pipeline Overview

The released SFT training data ([TOMATO-Star-SFT-Data-R1D-32B](https://huggingface.co/datasets/ZonglinY/TOMATO-Star-SFT-Data-R1D-32B)) and eval data ([TOMATO-Star](https://huggingface.co/datasets/ZonglinY/TOMATO-Star)) are available on HuggingFace. Download them first, then follow `main.sh` for the full pipeline with detailed documentation.

| Step | Name | Description |
|------|------|-------------|
| **1** | SFT Training | Supervised fine-tuning of the HC and IR models using LLaMA-Factory with DeepSpeed ZeRO-3 multi-node distributed training. |
| **2** | Evaluation | Evaluate HC quality via rubric scoring (Motivation, Mechanism, Methodology; 0--12 points) and IR accuracy on 15-way selection. Supports multi-node parallel evaluation. |
| **3** | Hierarchical Search Inference | Build a SPECTER2-clustered search tree, deploy the IR model with SGLang, and run best-first hierarchical search (O(log N)) or tournament search (O(N/K) baseline) over the inspiration corpus. |

## Quick Start

To use the released models for inference without reproducing the training pipeline:

### 1. Download Models

```bash
# Download from HuggingFace
huggingface-cli download ZonglinY/MOOSE-Star-HC-R1D-7B --local-dir ./models/MOOSE-Star-HC-R1D-7B
huggingface-cli download ZonglinY/MOOSE-Star-IR-R1D-7B --local-dir ./models/MOOSE-Star-IR-R1D-7B
```

### 2. Hypothesis Composition Inference

The HC model uses a fixed prompt template from training. Use `utils/prompt_store.py` to build the prompt correctly:

```python
import sys
sys.path.insert(0, "./utils")

from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt_store import instruction_prompts

model_path = "./models/MOOSE-Star-HC-R1D-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto", device_map="auto")

# Your inputs
research_question = "How can we improve drug-target interaction prediction?"
background_survey = "Existing methods rely on hand-crafted features and struggle to generalize..."
# For the first inspiration step: use this exact placeholder string (matches training format)
prev_hypothesis = "No previous hypothesis."
# For subsequent steps: pass the cumulative delta hypotheses from previous steps
inspiration_title = "Graph Neural Networks for Molecular Property Prediction"
inspiration_abstract = "We propose a GNN-based framework that learns molecular representations..."

# Build prompt using the exact training template
p = instruction_prompts("prepare_HC_sft_data_to_go_comprehensive_v2_delta")
prompt = (p[0] + research_question +
          p[1] + background_survey +
          p[2] + prev_hypothesis +
          p[3] + inspiration_title +
          p[4] + inspiration_abstract +
          p[5])

# Apply DeepSeek R1-Distill chat template (add_generation_prompt=False, then append manually)
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
formatted += "<｜Assistant｜>"

inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=8192, temperature=0.6, top_p=0.9)
# Decode only the newly generated tokens
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
# Output format:
# <think>... reasoning trace ...</think>
#
# Inspiration: [key concept from the inspiration paper]
# - Motivation (WHY): ...
# - Mechanism (HOW IT WORKS): ...
# - Methodology (HOW IT'S INTEGRATED): ...
```

### 3. Inspiration Retrieval Inference

The IR model is deployed via SGLang and queried through `Inference/ir_probability_extractor.py`, which returns a probability distribution over candidate papers. Install SGLang first:

```bash
pip install "sglang[all]>=0.5.0"
```

```python
import sys
sys.path.insert(0, "./Inference")
sys.path.insert(0, "./utils")

from ir_probability_extractor import IRProbabilityExtractor

# Deploy the IR model first (1 GPU sufficient for 7B model):
#   bash Inference/start_sglang.sh \
#       --model-path ./models/MOOSE-Star-IR-R1D-7B \
#       --port 1235        # tp=1 default; add --tp 2 for higher throughput
extractor = IRProbabilityExtractor(base_urls=["http://localhost:1235/v1"])

candidates = [
    {"title": "Graph Neural Networks for Molecular Property Prediction",
     "abstract": "We propose a GNN-based framework that learns molecular representations..."},
    {"title": "Attention Mechanisms in Sequence Models",
     "abstract": "We introduce multi-head self-attention for long-range dependency modeling..."},
    # ... up to 26 candidates; training used 15 (1 correct + 14 negatives)
]

result = extractor.get_selection_probabilities(
    research_question="How can we improve drug-target interaction prediction?",
    background_survey="Existing methods rely on hand-crafted features and struggle to generalize...",
    candidates=candidates,
    previous_hypothesis=None,  # None → "No previous hypothesis." internally
)
print(f"Selected candidate: {result.selected_label}")       # e.g., "A"
print(f"Probabilities: {result.probabilities}")             # {"A": 0.72, "B": 0.18, ...}
print(f"Selected index: {result.selected_index}")           # 0-indexed position in candidates list
```

### 4. Hierarchical Search with the IR Model

Deploy the IR model as an SGLang service and run full hierarchical search over the inspiration corpus:

```bash
# Step 1: Build the SPECTER2-clustered search tree (one-time)
bash Preprocessing/hierarchical_search/run_build_tree.sh

# Step 2: Start the SGLang inference server (1 GPU sufficient; use --tp 2 for higher throughput)
bash Inference/start_sglang.sh --model-path ./models/MOOSE-Star-IR-R1D-7B --port 1235

# Step 3: Run hierarchical search evaluation
python Inference/hierarchical_search_eval.py \
    --tree-dir ./data/hierarchical_search_tree \
    --eval-dir ./data/tomato_star/eval \
    --sglang-urls http://localhost:1235/v1 \
    --search-mode best_first \
    --max-samples 200 \
    --max-proposals 0 \
    --num-workers 200 \
    --softmax-temperature 5.0 \
    --max-tokens 20480 \
    --top-logprobs 30 \
    --output-dir ./data/search_eval_results
```

## Training

To reproduce the SFT training, first download the data from HuggingFace, then run the training scripts below.

```bash
# Download SFT training data (creates ./data/sft_data/HC/ and ./data/sft_data/IR/)
huggingface-cli download ZonglinY/TOMATO-Star-SFT-Data-R1D-32B --local-dir ./data/sft_data

# Download TOMATO-Star eval data
huggingface-cli download ZonglinY/TOMATO-Star --local-dir ./data/tomato_star

# Convert TOMATO-Star test.jsonl → individual JSON files (required by eval scripts)
# This is also included as Step 0 in main.sh
mkdir -p ./data/tomato_star/eval
python3 -c "
import json, os
with open('./data/tomato_star/test.jsonl') as f:
    for i, line in enumerate(f):
        paper = json.loads(line.strip())
        pid = paper.get('source_id', f'paper_{i}')
        with open(f'./data/tomato_star/eval/{pid}.json', 'w') as out:
            json.dump(paper, out, ensure_ascii=False, indent=2)
print('Done.')
"
```

Then configure the paths in `main.sh` and run the training scripts.

### Hypothesis Composition

```bash
# Full-parameter multi-node training
# Configure paths in SFT/full_train_hypothesis_composition.yaml, then:
bash SFT/train_hypothesis_composition.sh
```

Key training configuration (`SFT/full_train_hypothesis_composition.yaml`):
- Base model: `DeepSeek-R1-Distill-Qwen-7B`
- Template: `deepseekr1`
- Effective batch size: 128 (1 per device x 2 gradient accumulation x 64 GPUs)
- Learning rate: 1e-5 with cosine schedule
- 1 epoch, DeepSpeed ZeRO-3, bf16

### Inspiration Retrieval

```bash
# Full-parameter multi-node training
# Configure paths in SFT/full_train_inspiration_retrieval.yaml, then:
bash SFT/train_inspiration_retrieval.sh
```

Key training configuration (`SFT/full_train_inspiration_retrieval.yaml`):
- Base model: `DeepSeek-R1-Distill-Qwen-7B`
- Template: `deepseekr1`
- Effective batch size: 128 (1 per device x 1 gradient accumulation x 128 GPUs)
- Sequence length: 16384
- 1 epoch, DeepSpeed ZeRO-3, bf16

## Evaluation

### Hypothesis Composition Evaluation

Evaluates generated hypotheses using rubric scoring across three dimensions (Motivation, Mechanism, Methodology; each 0--4 points, total 0--12). Requires an LLM API endpoint for scoring.

```bash
# Normal evaluation (ground-truth inspirations)
bash Evaluation/run_hypothesis_composition_eval_rubric_parallel.sh

# Bounded evaluation (approximate inspirations, tests robustness)
bash Evaluation/run_hypothesis_composition_eval_rubric_bounded_parallel.sh
```

Configure `MODEL_PATH`, `OUTPUT_DIR`, and API settings at the top of each script. Supports multi-node parallel evaluation across multiple GPUs.

### Inspiration Retrieval Evaluation

Evaluates 15-way selection accuracy. Supports retry on extraction failure.

```bash
bash Evaluation/run_inspiration_retrieval_eval_parallel.sh
```

Configure `MODEL_PATH`, `DATA_FILE`, and `OUTPUT_DIR` at the top of the script.

## Project Structure

```
MOOSE-Star/
├── README.md
├── LICENSE                                          # Apache 2.0
├── requirements.txt
├── main.sh                                          # Full pipeline (Steps 1-3) with documentation
│
├── utils/
│   ├── common_utils.py                              # LLM client, text extraction utilities
│   ├── prompt_store.py                              # HC and IR prompt templates
│   ├── scoring_utils.py                             # Rubric scoring constants and parser
│   └── eval_utils.py                               # MDP road builder for eval
│
├── Preprocessing/
│   └── hierarchical_search/
│       ├── build_hierarchical_tree.py               # Build SPECTER2 search tree (Step 3, one-time)
│       ├── run_build_tree.sh
│       └── tree_search.py                           # Tree traversal utilities
│
├── SFT/
│   ├── full_train_hypothesis_composition.yaml       # HC full-param training config
│   ├── full_train_inspiration_retrieval.yaml        # IR full-param training config
│   ├── train_hypothesis_composition.sh              # HC multi-node launch script (Step 1)
│   ├── train_inspiration_retrieval.sh               # IR multi-node launch script (Step 1)
│   ├── deepspeed_zero2.json                         # ZeRO-2 config (optional alternative)
│   └── deepspeed_zero3.json                         # ZeRO-3 config (used by default)
│
├── Evaluation/
│   ├── hypothesis_composition_eval_rubric.py        # HC rubric evaluation (Step 2.1)
│   ├── hypothesis_composition_eval_rubric_bounded.py # HC bounded evaluation (Step 2.2)
│   ├── inspiration_retrieval_eval.py                # IR accuracy evaluation (Step 2.3)
│   ├── run_hypothesis_composition_eval_rubric_parallel.sh
│   ├── run_hypothesis_composition_eval_rubric_bounded_parallel.sh
│   └── run_inspiration_retrieval_eval_parallel.sh
│
└── Inference/
    ├── hierarchical_search_eval.py                  # Best-first hierarchical search (Step 3)
    ├── tournament_search_eval.py                    # Tournament search baseline (Step 3)
    ├── ir_probability_extractor.py                  # IR logprob extraction via SGLang
    ├── eval_utils.py
    └── start_sglang.sh                              # SGLang server launch script (Step 3)
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{yang2025moosestar,
  title={MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier},
  author={Yang, Zonglin and Bing, Lidong},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

## License

- **Code**: [Apache License 2.0](LICENSE)
- **Data** (TOMATO-Star, TOMATO-Star-SFT-Data-R1D-32B): [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
