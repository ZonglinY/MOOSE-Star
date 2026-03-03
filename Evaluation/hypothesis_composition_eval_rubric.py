"""
Hypothesis Composition Evaluation using LLM Scoring Rubric

- Single LLM call per sample for scoring
- Fixed 3 dimensions: Motivation (WHY), Mechanism (HOW IT WORKS), Methodology (HOW IT'S INTEGRATED)
- Score range: 0-12 total (0-4 per dimension)
- Reads hypothesis_components directly from sft_qa_data_dir
- Scoring is based on RECALL - what percentage of GT content is correctly covered.
  Both MISSING and WRONG content count as "not covered".
"""

import os
import sys
import json
import argparse
import time
import threading
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import PEFT for LoRA support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not installed. LoRA evaluation will not be available.")

# Add paths for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'utils'))

# Import from common_utils
from common_utils import (
    llm_generation,
    init_llm_client,
    extract_between_markers,
    extract_answer_content
)

# Import prompts for hypothesis generation
from prompt_store import instruction_prompts

# Import scoring rubric and utilities
from scoring_utils import (
    SCORING_RUBRIC,
    RERANKER_PROMPT_TEMPLATE,
    parse_scores
)

# Import MDP road builder (for dynamic generation from hypothesis_components)
from eval_utils import sample_one_MDP_for_one_paper_from_hypothesis_components

# Use reranker's template for evaluation (same format)
EVAL_PROMPT_TEMPLATE = RERANKER_PROMPT_TEMPLATE


# ============================================================================
# Thread-safe counter for parallel evaluation
# ============================================================================

class ThreadSafeCounter:
    """Thread-safe counter for tracking evaluations in parallel execution."""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self):
        with self._lock:
            return self._value
    
    def reset(self):
        with self._lock:
            self._value = 0


class HypothesisCompositionEvaluatorRubric:
    """
    Evaluator using LLM Scoring Rubric (single LLM call per sample).
    
    Scoring Dimensions:
    - Motivation (WHY): Does generated hypothesis identify the same research gap?
    - Mechanism (HOW IT WORKS): Does generated hypothesis propose the same core mechanism?
    - Methodology (HOW IT'S INTEGRATED): Does generated hypothesis describe similar implementation?
    
    Total Score: 0-12 (sum of three dimensions, each 0-4)
    """
    
    def __init__(
        self,
        model_path: str,
        lora_path: str = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
        max_length: int = 16384,
        max_new_tokens: int = 4096,
        sft_qa_data_dir: str = None,
        api_type: int = 0,
        api_key: str = "",
        base_url: str = "",
        model_name: str = "gpt-4o-mini",
        # Generation parameters (for hypothesis generation)
        temperature: float = 0.6,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        # Batch generation for better GPU utilization
        batch_size: int = 1,
        # Parallel LLM evaluation
        eval_max_workers: int = 16,
        # API max tokens for rubric evaluation
        api_max_tokens: int = 4096
    ):
        """
        Initialize the evaluator with a model and API client.
        
        Args:
            model_path: Path to base model (for generating hypothesis)
            lora_path: Path to LoRA checkpoint (None for base model evaluation)
            device: Device to use (cuda/cpu)
            load_in_8bit: Whether to load model in 8-bit precision
            max_length: Maximum sequence length
            max_new_tokens: Maximum new tokens to generate
            sft_qa_data_dir: Path to SFT QA data directory (contains hypothesis_components)
            api_type: API type (0: OpenAI-compatible)
            api_key: API key for evaluation
            base_url: Base URL for API
            model_name: Model name for API evaluation (recommend gpt-4o-mini)
            temperature: Generation temperature for hypothesis generation
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for token repetition
            batch_size: Batch size for hypothesis generation
            eval_max_workers: Number of parallel workers for LLM evaluation
            api_max_tokens: Maximum tokens for API evaluation output
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.sft_qa_data_dir = sft_qa_data_dir
        self.batch_size = batch_size
        self.eval_max_workers = eval_max_workers
        self.api_max_tokens = api_max_tokens
        
        # Initialize API client for evaluation
        self.api_type = api_type
        self.model_name = model_name
        self.client = init_llm_client(api_type, api_key, base_url)
        
        # Track failures (thread-safe for parallel evaluation)
        self.extraction_failures = ThreadSafeCounter()
        self.total_evaluations = ThreadSafeCounter()
        
        print(f"Loading model from {model_path}")
        if lora_path:
            print(f"Loading LoRA weights from {lora_path}")
        
        # Load tokenizer (use_fast=True for better performance)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with Flash Attention 2 for faster inference
        # Note: We unset WORLD_SIZE env var in bash script to prevent auto tp_plan="auto"
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",  # 2x faster attention
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        except Exception as e:
            print(f"Flash Attention 2 failed ({e}), falling back to default attention")
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        # Load LoRA weights if provided
        if lora_path:
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT is required for LoRA evaluation. Install with: pip install peft")
            
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                torch_dtype=torch.bfloat16
            )
            
            # Merge LoRA weights for faster inference
            self.model = self.model.merge_and_unload()
            print("LoRA weights loaded and merged")
        
        self.model.eval()
        print(f"Model loaded successfully")
        print(f"Evaluation API: {model_name} with {eval_max_workers} parallel workers")

    def _is_valid_mdp_step(self, delta_hyp: str) -> bool:
        """Check if an MDP step has valid delta hypothesis."""
        return delta_hyp is not None and delta_hyp.strip() != ""

    def generate_responses_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts in a batch for better GPU utilization.
        
        Args:
            prompts: List of user prompt contents (without system prompt or formatting)
            
        Returns:
            List of generated response texts (reasoning + hypothesis)
        """
        if not prompts:
            return []
        
        # Format all prompts using chat template
        # IMPORTANT: Use add_generation_prompt=False and manually add <｜Assistant｜>
        # This avoids adding <think>\n to prompt - model will generate <think>\n itself
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            formatted_prompt += "<｜Assistant｜>"
            formatted_prompts.append(formatted_prompt)
        
        # Tokenize with left padding for batch generation
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Use inference_mode for better performance than no_grad
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                num_beams=1,
                use_cache=True,  # Enable KV cache
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Batch decode is faster than loop decode
        max_input_len = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, max_input_len:]
        responses = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        return responses

    def _build_prev_hypothesis(self, mdp_road: List, step_idx: int) -> str:
        """Build prev_hypothesis by joining all previous delta hypotheses."""
        if step_idx == 0:
            return "No previous hypothesis."
        
        prev_deltas = [mdp_road[j][1] for j in range(step_idx)]
        return "\n\n".join(prev_deltas)

    def score_single_hypothesis(
        self, 
        gt_hypothesis: str, 
        generated_hypothesis: str,
        max_retries: int = 3
    ) -> Tuple[Optional[Dict[str, int]], Optional[float]]:
        """
        Score a single (GT, Generated) pair using LLM.
        
        Args:
            gt_hypothesis: Ground truth hypothesis
            generated_hypothesis: Generated hypothesis
            max_retries: Max retries for LLM call
            
        Returns:
            Tuple of (scores_dict, total_score)
            scores_dict: {'motivation': 0-4, 'mechanism': 0-4, 'methodology': 0-4}
            total_score: Sum (0-12)
            Returns (None, None) on failure
        """
        # Handle empty inputs
        if not generated_hypothesis or not generated_hypothesis.strip():
            return {'motivation': 0, 'mechanism': 0, 'methodology': 0}, 0.0
        
        if not gt_hypothesis or not gt_hypothesis.strip():
            return {'motivation': 0, 'mechanism': 0, 'methodology': 0}, 0.0
        
        # Build prompt
        prompt = EVAL_PROMPT_TEMPLATE.format(
            gt_hypothesis=gt_hypothesis,
            generated_hypothesis=generated_hypothesis,
            scoring_rubric=SCORING_RUBRIC
        )
        
        # Call LLM with retries
        for attempt in range(max_retries):
            try:
                response = llm_generation(
                    prompt,
                    self.model_name,
                    self.client,
                    temperature=0.0,  # Deterministic for evaluation
                    api_type=self.api_type,
                    max_tokens=self.api_max_tokens
                )
                
                scores = parse_scores(response)
                if scores:
                    total = scores['motivation'] + scores['mechanism'] + scores['methodology']
                    return scores, float(total)
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                else:
                    print(f"Warning: Failed to score after {max_retries} attempts: {e}")
        
        return None, None

    def evaluate_single_step(
        self, 
        gt_hypothesis: str,
        pre_generated_response: str
    ) -> Dict:
        """
        Evaluate a single step using rubric scoring.
        
        Args:
            gt_hypothesis: Ground truth hypothesis (delta_hypothesis from hypothesis_components)
            pre_generated_response: Pre-generated model response
            
        Returns:
            Dict with evaluation results
        """
        # Validate gt_hypothesis
        if not self._is_valid_mdp_step(gt_hypothesis):
            return None
        
        # Validate pre_generated_response (defensive check)
        if pre_generated_response is None:
            print(f"Warning: pre_generated_response is None for valid step")
            return None
        
        # Extract reasoning trace (content between <think> and </think>)
        think_start_pos = pre_generated_response.find('<think>')
        think_end_pos = pre_generated_response.find('</think>')
        if think_end_pos != -1:
            # Extract content between <think> and </think>
            start_pos = think_start_pos + len('<think>') if think_start_pos != -1 else 0
            reasoning_trace = pre_generated_response[start_pos:think_end_pos].strip()
        else:
            # No </think> found, use full response (might not be R1 format)
            reasoning_trace = pre_generated_response
        
        # Extract generated hypothesis
        # Try v2 delta format: **Delta Hypothesis starts:** ... **Delta Hypothesis ends**
        delta_hyp = extract_between_markers(pre_generated_response, r'Delta\s*Hypothesis')
        if delta_hyp:
            gene_hyp = delta_hyp.strip()
        else:
            # Fallback to extract_answer_content
            gene_hyp = extract_answer_content(pre_generated_response)
        
        # Score using rubric (single LLM call)
        self.total_evaluations.increment()
        scores_dict, total_score = self.score_single_hypothesis(gt_hypothesis, gene_hyp)
        
        if scores_dict is None:
            self.extraction_failures.increment()
            print(f"Warning: Score extraction failed")
            return {
                'scores': None,
                'total_score': None,
                'generated_hypothesis': gene_hyp,
                'reasoning_trace': reasoning_trace,
                'ground_truth_hypothesis': gt_hypothesis,
                'extraction_failed': True
            }
        
        return {
            'scores': scores_dict,
            'total_score': total_score,
            'generated_hypothesis': gene_hyp,
            'reasoning_trace': reasoning_trace,
            'ground_truth_hypothesis': gt_hypothesis,
            'extraction_failed': False
        }

    def evaluate_eval_dataset(self, output_path: str = None):
        """
        Evaluate the model on the full eval dataset.
        
        Uses PIPELINED execution for maximum GPU utilization:
        - Generation and evaluation run in parallel
        - As soon as a batch is generated, it's queued for evaluation
        
        Args:
            output_path: Path to save evaluation results (will be treated as a folder)
            
        Returns:
            List of evaluation results
        """
        import queue
        import threading
        
        # Get evaluation files from sft_qa_data_dir
        eval_files = [f for f in os.listdir(self.sft_qa_data_dir) if f.endswith('.json')]
        
        if not eval_files:
            print(f"Warning: No JSON files found in {self.sft_qa_data_dir}")
            return []
        
        print(f"Found {len(eval_files)} JSON files to evaluate")
        
        # ========== Phase 1: Collect all generation tasks from all files ==========
        print("\n[Phase 1] Collecting generation tasks from all files...")
        print("  → Reading directly from sft_qa_data_dir")
        gen_prompts = instruction_prompts("prepare_HC_sft_data_to_go_comprehensive_v2_delta")
        
        all_gen_tasks = []  # (file_name, step_idx, gt_hypothesis, prompt)
        skipped_files = 0
        
        for cur_file in tqdm(eval_files, desc="Reading files"):
            # Load SFT QA data
            cur_sft_qa_data_file_path = os.path.join(self.sft_qa_data_dir, cur_file)
            if not os.path.exists(cur_sft_qa_data_file_path):
                skipped_files += 1
                continue
            
            with open(cur_sft_qa_data_file_path, "r") as f:
                cur_sft_qa_data = json.load(f)
            
            research_question = cur_sft_qa_data["research_question"]
            background_survey = cur_sft_qa_data["background_survey"]
            inspirations = cur_sft_qa_data["inspiration"]
            hypothesis_components = cur_sft_qa_data["hypothesis_components"]
            
            if not research_question or not inspirations or not hypothesis_components:
                skipped_files += 1
                continue
            
            # Dynamically build MDP road from hypothesis_components
            try:
                mdp_road = sample_one_MDP_for_one_paper_from_hypothesis_components(
                    inspirations, hypothesis_components, cur_file
                )
            except Exception as e:
                print(f"Warning: Failed to build MDP road for {cur_file}: {e}")
                skipped_files += 1
                continue
            
            # Build prompts for valid MDP steps
            for step_idx, (insp_id, delta_hyp) in enumerate(mdp_road):
                if not self._is_valid_mdp_step(delta_hyp):
                    print(f"Warning: Invalid MDP step for {cur_file}, step_idx: {step_idx}, delta_hyp: {delta_hyp}")
                    continue
                
                # Get inspiration info
                cur_insp = inspirations[insp_id]
                found_title = cur_insp["found_title"]
                found_abstract = cur_insp["found_abstract"]
                
                # Build prev_hyp (cumulative of all previous deltas)
                prev_hyp = self._build_prev_hypothesis(mdp_road, step_idx)
                
                prompt = (gen_prompts[0] + research_question + gen_prompts[1] + background_survey + 
                         gen_prompts[2] + prev_hyp + gen_prompts[3] + found_title + 
                         gen_prompts[4] + found_abstract + gen_prompts[5])
                
                # Store gt_hypothesis directly (delta_hyp from hypothesis_components)
                all_gen_tasks.append((cur_file, step_idx, delta_hyp, prompt))
        
        if skipped_files > 0:
            print(f"  Skipped {skipped_files} files (missing required fields)")
        
        total_tasks = len(all_gen_tasks)
        print(f"Total generation tasks: {total_tasks}")
        
        if not all_gen_tasks:
            print("No generation tasks found.")
            return []
        
        # ========== Phase 2: Pipelined Generation + Evaluation ==========
        print(f"\n[Phase 2] Pipelined execution (batch_size={self.batch_size}, eval_workers={self.eval_max_workers})...")
        print("  → Generation and evaluation running in parallel")
        
        # Queue for passing generated results to evaluation
        eval_queue = queue.Queue(maxsize=self.eval_max_workers * 2)
        eval_results = []
        eval_results_lock = threading.Lock()
        generation_done = threading.Event()
        
        # Progress tracking
        generated_count = [0]
        evaluated_count = [0]
        
        def eval_worker():
            """Worker thread that evaluates generated responses."""
            while True:
                try:
                    item = eval_queue.get(timeout=1.0)
                    if item is None:  # Poison pill
                        break
                    
                    task_idx, (file_name, step_idx, gt_hypothesis, prompt), response = item
                    
                    # Evaluate using rubric scoring
                    result = self.evaluate_single_step(gt_hypothesis, response)
                    
                    if result:
                        result['file'] = file_name
                        result['step_idx'] = step_idx
                        
                        with eval_results_lock:
                            eval_results.append(result)
                            evaluated_count[0] += 1
                    
                    eval_queue.task_done()
                    
                except queue.Empty:
                    if generation_done.is_set() and eval_queue.empty():
                        break
                    continue
        
        # Start evaluation workers
        eval_threads = []
        for _ in range(self.eval_max_workers):
            t = threading.Thread(target=eval_worker, daemon=True)
            t.start()
            eval_threads.append(t)
        
        # Generation loop (main thread) - feeds eval queue
        prompts = [t[3] for t in all_gen_tasks]
        pbar = tqdm(total=total_tasks, desc="Gen→Eval")
        
        for batch_start in range(0, len(prompts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            
            # Generate batch
            batch_responses = self.generate_responses_batch(batch_prompts)
            generated_count[0] += len(batch_responses)
            
            # Queue for evaluation (will block if queue is full - backpressure)
            for i, response in enumerate(batch_responses):
                task_idx = batch_start + i
                eval_queue.put((task_idx, all_gen_tasks[task_idx], response))
            
            # Update progress
            pbar.n = evaluated_count[0]
            pbar.set_postfix({'gen': generated_count[0], 'eval': evaluated_count[0]})
            pbar.refresh()
        
        # Signal generation complete
        generation_done.set()
        
        # Wait for all evaluations to complete
        eval_queue.join()
        
        # Send poison pills to stop workers
        for _ in eval_threads:
            eval_queue.put(None)
        for t in eval_threads:
            t.join(timeout=5.0)
        
        pbar.n = evaluated_count[0]
        pbar.close()
        
        print(f"\nCompleted: Generated {generated_count[0]}, Evaluated {evaluated_count[0]}")
        
        # Calculate overall metrics
        print("\n" + "="*60)
        print("EVALUATION METRICS SUMMARY (Rubric Scoring)")
        print("="*60)
        
        # Extract scores (excluding failures)
        all_total_scores = []
        all_motivation_scores = []
        all_mechanism_scores = []
        all_methodology_scores = []
        all_hypothesis_lengths = []
        
        for result in eval_results:
            if isinstance(result, dict) and not result.get('extraction_failed', False):
                # Both total_score AND scores must be valid
                if result.get('total_score') is not None and result.get('scores') is not None:
                    all_total_scores.append(result['total_score'])
                    scores = result['scores']  # Direct access since we verified it's not None
                    all_motivation_scores.append(scores['motivation'])
                    all_mechanism_scores.append(scores['mechanism'])
                    all_methodology_scores.append(scores['methodology'])
                
                if result.get('generated_hypothesis'):
                    word_count = len(result['generated_hypothesis'].split())
                    all_hypothesis_lengths.append(word_count)
        
        mean_total = None
        mean_motivation = None
        mean_mechanism = None
        mean_methodology = None
        
        if all_total_scores:
            mean_total = sum(all_total_scores) / len(all_total_scores)
            mean_motivation = sum(all_motivation_scores) / len(all_motivation_scores)
            mean_mechanism = sum(all_mechanism_scores) / len(all_mechanism_scores)
            mean_methodology = sum(all_methodology_scores) / len(all_methodology_scores)
            
            print(f"\nOverall Mean Total Score: {mean_total:.2f} / 12")
            print(f"  - This is the PRIMARY METRIC for model comparison")
            print(f"  - Higher is better (range: 0 to 12)")
            print(f"\nDimension Scores (0-4 each):")
            print(f"  Motivation (WHY):           {mean_motivation:.2f}")
            print(f"  Mechanism (HOW IT WORKS):   {mean_mechanism:.2f}")
            print(f"  Methodology (INTEGRATION):  {mean_methodology:.2f}")
            print(f"\nStatistics:")
            print(f"  Min total: {min(all_total_scores):.2f}")
            print(f"  Max total: {max(all_total_scores):.2f}")
            print(f"  Valid evaluations: {len(all_total_scores)}")
            print(f"  Total samples processed: {len(eval_results)}")
        else:
            print("No valid scores calculated.")
        
        avg_hypothesis_length = None
        if all_hypothesis_lengths:
            avg_hypothesis_length = sum(all_hypothesis_lengths) / len(all_hypothesis_lengths)
            print(f"\nAverage generated hypothesis length: {avg_hypothesis_length:.1f} words")
        
        # Report extraction failures
        total_evals = self.total_evaluations.value
        extraction_fails = self.extraction_failures.value
        if extraction_fails > 0:
            print("\n" + "="*60)
            print("EXTRACTION FAILURE SUMMARY")
            print("="*60)
            print(f"Total evaluations attempted: {total_evals}")
            print(f"Extraction failures: {extraction_fails}")
            print(f"Success rate: {(total_evals - extraction_fails)/total_evals:.2%}")
        elif total_evals > 0:
            print(f"\n✓ All {total_evals} evaluations completed successfully.")
        
        # Save results
        if output_path:
            output_folder = output_path.rstrip('.json')
            os.makedirs(output_folder, exist_ok=True)
            
            # 1. metrics.json - successful evaluations with scores
            metrics_data = []
            for result in eval_results:
                # Only include results with valid scores (not failed extractions)
                if not result.get('extraction_failed', False) and result.get('scores') is not None:
                    metrics_entry = {
                        'file': result['file'],
                        'step_idx': result.get('step_idx', 0),
                        'scores': result['scores'],
                        'total_score': result['total_score']
                    }
                    metrics_data.append(metrics_entry)
            
            metrics_path = os.path.join(output_folder, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            print(f"\nMetrics saved to {metrics_path} ({len(metrics_data)} evaluations)")
            
            # 2. generations.json - ALL results including failures
            generations_data = []
            for result in eval_results:
                gen_entry = {
                    'file': result['file'],
                    'step_idx': result.get('step_idx', 0),
                    'scores': result.get('scores'),
                    'total_score': result.get('total_score'),
                    'generated_hypothesis': result.get('generated_hypothesis', ''),
                    'reasoning_trace': result.get('reasoning_trace', ''),
                    'ground_truth_hypothesis': result.get('ground_truth_hypothesis', ''),
                    'extraction_failed': result.get('extraction_failed', False)
                }
                generations_data.append(gen_entry)
            
            generations_path = os.path.join(output_folder, 'generations.json')
            with open(generations_path, 'w') as f:
                json.dump(generations_data, f, indent=2)
            failed_count = sum(1 for r in eval_results if r.get('extraction_failed', False))
            print(f"Generations saved to {generations_path} ({len(generations_data)} total, {failed_count} failures)")
            
            # 3. summary.json - overall statistics
            summary_data = {
                'mean_total_score': mean_total,
                'mean_motivation_score': mean_motivation,
                'mean_mechanism_score': mean_mechanism,
                'mean_methodology_score': mean_methodology,
                'min_total_score': min(all_total_scores) if all_total_scores else None,
                'max_total_score': max(all_total_scores) if all_total_scores else None,
                'average_hypothesis_length': avg_hypothesis_length,
                'total_evaluations': len(all_total_scores),
                'total_samples_processed': len(eval_results),
                'extraction_failures': extraction_fails,
                'total_evaluations_attempted': total_evals
            }
            
            summary_path = os.path.join(output_folder, 'summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            print(f"Summary saved to {summary_path}")
            
            print(f"\nAll results saved to folder: {output_folder}")
        
        return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate hypothesis composition using rubric scoring')
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA checkpoint (optional)")
    
    # Evaluation settings
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--max_length", type=int, default=16384, help="Maximum sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum new tokens to generate")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.6, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    
    # API settings for LLM evaluation
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", 
                       help="Model name for API evaluation (recommend gpt-4o-mini)")
    parser.add_argument("--api_type", type=int, default=0, help="0: OpenAI-compatible")
    parser.add_argument("--api_key", type=str, default="", help="API key")
    parser.add_argument("--base_url", type=str, default="", help="Base URL for API")
    parser.add_argument("--eval_max_workers", type=int, default=16, 
                       help="Number of parallel workers for LLM evaluation")
    parser.add_argument("--api_max_tokens", type=int, default=4096,
                       help="Maximum tokens for API evaluation output")
    
    # Dataset paths
    parser.add_argument("--sft_qa_data_dir", type=str, required=True, 
                       help="Path to SFT QA data directory (contains hypothesis_components)")
    
    # Output path
    parser.add_argument("--eval_result_path", type=str, required=True, 
                       help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = HypothesisCompositionEvaluatorRubric(
        model_path=args.model_path,
        lora_path=args.lora_path,
        device="cuda",
        load_in_8bit=args.load_in_8bit,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        sft_qa_data_dir=args.sft_qa_data_dir,
        api_type=args.api_type,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        batch_size=args.batch_size,
        eval_max_workers=args.eval_max_workers,
        api_max_tokens=args.api_max_tokens
    )
    
    # Run evaluation
    evaluator.evaluate_eval_dataset(output_path=args.eval_result_path)

