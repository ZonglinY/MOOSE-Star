"""
Scoring utilities for Hypothesis Composition evaluation.

Contains the rubric, prompt template, and score parser used in:
- Evaluation/hypothesis_composition_eval_rubric.py
- Evaluation/hypothesis_composition_eval_rubric_bounded.py
"""

import re
import json
from typing import Dict, Optional

from common_utils import extract_between_markers


# ============================================================================
# Scoring Rubric (5-point scale per dimension)
# ============================================================================

SCORING_RUBRIC = """
## Scoring Rubric (0-4 for each dimension)

**IMPORTANT INSTRUCTIONS:**
1. Score based on RECALL - what percentage of GT content is correctly covered by Generated.
2. Both MISSING and WRONG content count as "not covered".
3. The examples below are ONLY for illustration - do NOT match against them.
   Score the actual GT vs Generated content, not similarity to examples.

---

### Motivation (WHY) - Does it identify the same research gap?

**How to score:** Count what percentage of GT's key elements appear correctly in Generated.

[EXAMPLE FOR ILLUSTRATION ONLY - do not match against this]
Example GT: "Current deep learning methods for brain tumor segmentation in MRI scans suffer from
     low accuracy at tumor boundaries, particularly in low-contrast glioma regions,
     due to insufficient modeling of boundary uncertainty"
(5 key elements: domain=brain tumor/MRI, task=segmentation, problem=low boundary accuracy,
 context=low-contrast glioma, cause=insufficient uncertainty modeling)

- 4 (~100%): "Existing DL approaches for brain tumor MRI segmentation have poor boundary
              delineation in glioma cases because they fail to model boundary uncertainty"
             ✓ All 5: domain + task + problem + context + cause
- 3 (~75%):  Missing: "Brain tumor segmentation methods have low accuracy at boundaries in gliomas"
             ✓ 4: domain + task + problem + context | ✗ Missing: cause
             Wrong: "Brain tumor segmentation in MRI has low boundary accuracy in gliomas due to limited training data"
             ✓ 4: domain + task + problem + context | ✗ Wrong: cause (limited data vs uncertainty modeling)
- 2 (~50%):  Missing: "Brain tumor segmentation in MRI has accuracy issues"
             ✓ 2.5: domain + task + vague problem | ✗ Missing: context + cause
             Wrong: "Brain tumor detection in MRI suffers from false positives in gliomas"
             ✓ 2: domain + context | ✗ Wrong: task (detection) + problem (false positives)
- 1 (~25%):  Missing: "Medical image segmentation needs improvement"
             ✓ 1: broad domain only | ✗ Missing: specific target + problem + context + cause
             Wrong: "Brain tumor classification methods are too slow"
             ✓ 1: target organ | ✗ Wrong: task (classification) + problem (speed)
- 0 (~0%):   "Protein structure prediction lacks accuracy"
             ✓ 0 | ✗ Completely unrelated domain
             "Natural language models struggle with long-range dependencies"
             ✓ 0 | ✗ Completely unrelated domain

### Mechanism (HOW IT WORKS) - Does it propose the same core mechanism?
GT: "Apply transformer-based attention with boundary-aware loss functions to learn
     multi-scale feature representations, enabling precise tumor boundary localization
     through uncertainty-guided refinement"
(5 key elements: architecture=transformer attention, loss=boundary-aware,
 features=multi-scale, task=boundary localization, technique=uncertainty refinement)

- 4 (~100%): "Use transformer attention with boundary loss for multi-scale features
              to localize tumor boundaries via uncertainty-guided refinement"
             ✓ All 5: architecture + loss + features + task + technique
- 3 (~75%):  Missing: "Transformer attention with boundary-aware loss for multi-scale
              feature learning to localize tumor boundaries"
             ✓ 4: architecture + loss + features + task | ✗ Missing: refinement technique
             Wrong: "Transformer attention with boundary loss for multi-scale features
              to localize boundaries via post-processing CRF"
             ✓ 4: architecture + loss + features + task | ✗ Wrong: technique (CRF vs uncertainty)
- 2 (~50%):  Missing: "Use attention mechanism with boundary loss for tumor boundary detection"
             ✓ 2.5: partial architecture + loss + task | ✗ Missing: multi-scale + refinement
             Wrong: "Use transformer attention with standard loss for multi-scale feature learning"
             ✓ 2.5: architecture + features | ✗ Wrong: loss (standard vs boundary) + missing: task + technique
- 1 (~25%):  Missing: "Apply deep learning for tumor analysis"
             ✓ 0.5: broad method category | ✗ Missing: specific architecture + loss + features + technique
             Wrong: "Use transformer attention for image classification"
             ✓ 1: architecture | ✗ Wrong: task (classification) + missing: loss + features + technique
- 0 (~0%):   "Apply LSTM for time series forecasting"
             ✓ 0 | ✗ Completely unrelated mechanism and task
             "Use rule-based heuristics for text classification"
             ✓ 0 | ✗ Completely unrelated mechanism and domain

### Methodology (HOW IT'S INTEGRATED) - Does it describe similar implementation?
GT: "Train on BraTS 2021 dataset (1251 MRI cases), implement 3D U-Net with transformer
     encoder, use combined Dice-boundary loss, apply 5-fold cross-validation,
     evaluate with Dice score and 95% Hausdorff distance"
(6 key details: dataset=BraTS 2021/1251, architecture=3D U-Net + transformer,
 loss=Dice-boundary, validation=5-fold CV, metrics=Dice + HD95)

- 4 (~100%): "Train on BraTS 2021 (1251 scans), 3D U-Net with transformer encoder,
              Dice-boundary loss, 5-fold CV, report Dice and HD95"
             ✓ All 6: dataset + architecture + loss + validation + metrics
- 3 (~75%):  Missing: "BraTS 2021 dataset, 3D U-Net + transformer, Dice-boundary loss,
              5-fold CV, Dice score"
             ✓ 5: dataset + architecture + loss + validation + partial metrics | ✗ Missing: HD95
             Wrong: "BraTS 2021 (1251 scans), 3D U-Net + transformer, Dice-boundary loss,
              3-fold CV, Dice and HD95"
             ✓ 5: dataset + architecture + loss + metrics | ✗ Wrong: validation (3-fold vs 5-fold)
- 2 (~50%):  Missing: "Train on BraTS dataset with 3D U-Net, evaluate Dice score"
             ✓ 3: partial dataset + architecture + partial metrics | ✗ Missing: size + loss + validation + HD95
             Wrong: "Train on private dataset, use 2D U-Net, Dice-boundary loss, 5-fold CV, Dice"
             ✓ 3: loss + validation + partial metrics | ✗ Wrong: dataset + architecture (2D vs 3D)
- 1 (~25%):  Missing: "Train a segmentation model on brain MRI data"
             ✓ 1.5: vague dataset + vague approach | ✗ Missing: specific details
             Wrong: "Train on TCGA genomic data, use ResNet, cross-entropy, accuracy"
             ✓ 0.5: general training | ✗ Wrong: dataset + architecture + loss + metrics
- 0 (~0%):   "Fine-tune GPT on dialogue dataset with RLHF"
             ✓ 0 | ✗ Completely different domain and methodology
             "Survey 200 patients using questionnaires, analyze with chi-square test"
             ✓ 0 | ✗ Completely different methodology type
"""


RERANKER_PROMPT_TEMPLATE = """You are evaluating how well a Generated Hypothesis matches a Ground Truth (GT) Hypothesis.

## Task
For each dimension, count what percentage of GT's key elements are CORRECTLY covered by Generated.
- MISSING content = not covered
- WRONG content = not covered (counts same as missing)
- Only CORRECT matches count toward coverage

## Ground Truth Hypothesis (the reference):
{gt_hypothesis}

## Generated Hypothesis (to be scored):
{generated_hypothesis}

{scoring_rubric}

## Scoring Process:
1. For each dimension (Motivation/Mechanism/Methodology):
   a. Identify the key elements in the GT
   b. Check which elements appear CORRECTLY in Generated
   c. Calculate coverage percentage → map to score (0-4)
2. Be strict: 4 requires ~100% correct coverage
3. Empty or irrelevant responses → 0

## Output Format (MUST follow exactly):

**Motivation Score starts:** [0-4] **Motivation Score ends**
**Mechanism Score starts:** [0-4] **Mechanism Score ends**
**Methodology Score starts:** [0-4] **Methodology Score ends**
"""


def parse_scores(response: str) -> Optional[Dict[str, int]]:
    """Parse LLM response to extract scores using starts/ends markers."""
    scores = {}

    # Extract each score using markers
    for field in ['Motivation Score', 'Mechanism Score', 'Methodology Score']:
        key = field.split()[0].lower()  # "motivation", "mechanism", "methodology"
        value = extract_between_markers(response, field)
        if value:
            # Extract numeric value
            num_match = re.search(r'(\d)', value)
            if num_match:
                scores[key] = max(0, min(4, int(num_match.group(1))))

    # Check if all three scores are extracted
    if len(scores) == 3:
        return scores

    # Fallback: try JSON format for backward compatibility
    json_match = re.search(r'\{[^}]+\}', response)
    if json_match:
        try:
            json_scores = json.loads(json_match.group())
            if all(k in json_scores for k in ['motivation', 'mechanism', 'methodology']):
                for k in ['motivation', 'mechanism', 'methodology']:
                    json_scores[k] = max(0, min(4, int(json_scores[k])))
                return json_scores
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: try to extract individual scores with regex
    try:
        motivation = int(re.search(r'motivation["\s:]+(\d)', response, re.I).group(1))
        mechanism = int(re.search(r'mechanism["\s:]+(\d)', response, re.I).group(1))
        methodology = int(re.search(r'methodology["\s:]+(\d)', response, re.I).group(1))
        return {
            'motivation': max(0, min(4, motivation)),
            'mechanism': max(0, min(4, mechanism)),
            'methodology': max(0, min(4, methodology))
        }
    except (AttributeError, ValueError):
        return None
