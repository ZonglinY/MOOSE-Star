"""
Hypothesis Composition Evaluation for Bounded Composition (Rubric Scoring)

This script evaluates hypothesis composition under imperfect retrieval conditions.
Instead of using ground truth inspirations, it uses bounded inspirations 
(semantically similar but not exact) from different difficulty tiers.

Key differences from hypothesis_composition_eval_rubric.py:
- Uses bounded_selections_dir with pre-selected bounded inspirations
- Evaluates separately for each tier (hard, medium, easy)
- Reports tier-wise statistics to understand robustness to retrieval quality

Tier Definitions (similarity to GT inspiration):
- hard:   [0.90, 0.92) - Most different from GT
- medium: [0.92, 0.94)
- easy:   [0.94, 0.97) - Most similar to GT
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

EVAL_PROMPT_TEMPLATE = RERANKER_PROMPT_TEMPLATE

# Tier definitions (same as bounded_inspiration_selector_worker_v2.py)
TIERS = ['hard', 'medium', 'easy']


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


class BoundedCompositionEvaluator:
    """
    Evaluator for Bounded Composition using LLM Scoring Rubric.
    
    Evaluates model performance when given imperfect (bounded) inspirations
    instead of ground truth inspirations.
    """
    
    def __init__(
        self,
        model_path: str,
        lora_path: str = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
        max_length: int = 16384,
        max_new_tokens: int = 8192,
        bounded_selections_dir: str = None,
        tiers: List[str] = None,
        api_type: int = 0,
        api_key: str = "",
        base_url: str = "",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.6,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        batch_size: int = 1,
        eval_max_workers: int = 16,
        api_max_tokens: int = 16384,
        max_files: int = None
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.bounded_selections_dir = bounded_selections_dir
        self.tiers = tiers or TIERS
        self.batch_size = batch_size
        self.eval_max_workers = eval_max_workers
        self.api_max_tokens = api_max_tokens
        self.max_files = max_files
        
        # Initialize API client for evaluation
        self.api_type = api_type
        self.model_name = model_name
        self.client = init_llm_client(api_type, api_key, base_url)
        
        # Track failures (thread-safe)
        self.extraction_failures = ThreadSafeCounter()
        self.total_evaluations = ThreadSafeCounter()
        
        # Get prompts
        self.gen_prompts = instruction_prompts("prepare_HC_sft_data_to_go_comprehensive_v2_delta")
        
        print(f"Loading model from {model_path}")
        if lora_path:
            print(f"Loading LoRA weights from {lora_path}")
        
        # Load tokenizer (use_fast=True for better performance)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Note: For data parallel evaluation, distributed env vars should be set
        # in the launch script to prevent tensor parallelism initialization
        
        # Load model with Flash Attention 2 for faster inference
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "tp_plan": None,
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
                raise ImportError("PEFT is required for LoRA evaluation.")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                torch_dtype=torch.bfloat16
            )
            self.model = self.model.merge_and_unload()
            print("LoRA weights loaded and merged")
        
        self.model.eval()
        print(f"Model loaded successfully")
        print(f"Evaluation tiers: {self.tiers}")
        print(f"Evaluation API: {model_name} with {eval_max_workers} parallel workers")

    def generate_responses_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for multiple prompts in a batch."""
        if not prompts:
            return []
        
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            formatted_prompt += "<｜Assistant｜>"
            formatted_prompts.append(formatted_prompt)
        
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

    def build_prompt(
        self,
        research_question: str,
        background_survey: str,
        prev_hypothesis: str,
        inspiration_title: str,
        inspiration_abstract: str
    ) -> str:
        """Build generation prompt with bounded inspiration."""
        return (
            self.gen_prompts[0] + research_question +
            self.gen_prompts[1] + background_survey +
            self.gen_prompts[2] + prev_hypothesis +
            self.gen_prompts[3] + inspiration_title +
            self.gen_prompts[4] + inspiration_abstract +
            self.gen_prompts[5]
        )

    def score_single_hypothesis(
        self,
        gt_hypothesis: str,
        generated_hypothesis: str,
        max_retries: int = 3
    ) -> Tuple[Optional[Dict[str, int]], Optional[float]]:
        """Score a single (GT, Generated) pair using LLM."""
        if not generated_hypothesis or not generated_hypothesis.strip():
            return {'motivation': 0, 'mechanism': 0, 'methodology': 0}, 0.0
        
        if not gt_hypothesis or not gt_hypothesis.strip():
            return {'motivation': 0, 'mechanism': 0, 'methodology': 0}, 0.0
        
        prompt = EVAL_PROMPT_TEMPLATE.format(
            gt_hypothesis=gt_hypothesis,
            generated_hypothesis=generated_hypothesis,
            scoring_rubric=SCORING_RUBRIC
        )
        
        for attempt in range(max_retries):
            try:
                response = llm_generation(
                    prompt,
                    self.model_name,
                    self.client,
                    temperature=0.0,
                    api_type=self.api_type,
                    max_tokens=self.api_max_tokens
                )
                
                scores = parse_scores(response)
                if scores:
                    total = scores['motivation'] + scores['mechanism'] + scores['methodology']
                    return scores, float(total)
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                else:
                    print(f"Warning: Failed to score after {max_retries} attempts: {e}")
        
        return None, None

    def extract_hypothesis_from_response(self, response: str) -> str:
        """Extract hypothesis from model response."""
        # Try v2 delta format first
        delta_hyp = extract_between_markers(response, r'Delta\s*Hypothesis')
        if delta_hyp:
            return delta_hyp.strip()
        # Fallback
        return extract_answer_content(response)

    def _process_single_file(self, cur_file: str) -> List[Dict]:
        """
        Process a single bounded selections file.
        
        For each inspiration with bounded_selections, generates hypothesis
        using bounded inspiration (per tier) and evaluates against GT.
        """
        file_results = []
        
        file_path = os.path.join(self.bounded_selections_dir, cur_file)
        if not os.path.exists(file_path):
            return file_results
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        research_question = data.get('research_question', '')
        background_survey = data.get('background_survey', '')
        inspirations = data.get('inspirations', [])
        
        if not research_question or not inspirations:
            return file_results
        
        # Collect all generation tasks
        gen_tasks = []  # (idx, tier, prompt, gt_hypothesis, bounded_info)
        
        prev_hypothesis = "No previous hypothesis."
        
        for insp in inspirations:
            idx = insp.get('idx', 0)
            gt_hypothesis = insp.get('delta_hypothesis', '')
            bounded_selections = insp.get('bounded_selections', {})
            gt_title = insp.get('gt_title', '')
            
            if not gt_hypothesis:
                # Update prev_hypothesis anyway for continuity
                continue
            
            # Generate for each tier
            for tier in self.tiers:
                bounded = bounded_selections.get(tier)
                if bounded is None:
                    continue
                
                bounded_title = bounded.get('title', '')
                bounded_abstract = bounded.get('abstract', '')
                similarity = bounded.get('similarity', 0)
                
                if not bounded_title or not bounded_abstract:
                    continue
                
                prompt = self.build_prompt(
                    research_question,
                    background_survey,
                    prev_hypothesis,
                    bounded_title,
                    bounded_abstract
                )
                
                bounded_info = {
                    'bounded_title': bounded_title,
                    'bounded_similarity': similarity,
                    'gt_inspiration_title': gt_title
                }
                
                gen_tasks.append((idx, tier, prompt, gt_hypothesis, bounded_info))
            
            # Update prev_hypothesis for next step
            if gt_hypothesis:
                if prev_hypothesis == "No previous hypothesis.":
                    prev_hypothesis = gt_hypothesis
                else:
                    prev_hypothesis = prev_hypothesis + "\n\n" + gt_hypothesis
        
        if not gen_tasks:
            return file_results
        
        # Batch generate responses
        prompts = [t[2] for t in gen_tasks]
        all_responses = []
        for batch_start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[batch_start:batch_start + self.batch_size]
            batch_responses = self.generate_responses_batch(batch_prompts)
            all_responses.extend(batch_responses)
        
        # Prepare evaluation tasks
        eval_tasks = []
        for i, (idx, tier, prompt, gt_hypothesis, bounded_info) in enumerate(gen_tasks):
            response = all_responses[i]
            eval_tasks.append((idx, tier, response, gt_hypothesis, bounded_info))
        
        # Parallel LLM evaluation
        def eval_task(args):
            idx, tier, response, gt_hypothesis, bounded_info = args
            
            self.total_evaluations.increment()
            
            # Extract generated hypothesis
            gene_hyp = self.extract_hypothesis_from_response(response)
            
            # Extract reasoning trace
            think_start = response.find('<think>')
            think_end = response.find('</think>')
            if think_end != -1:
                start_pos = think_start + len('<think>') if think_start != -1 else 0
                reasoning_trace = response[start_pos:think_end].strip()
            else:
                reasoning_trace = response
            
            # Score
            scores_dict, total_score = self.score_single_hypothesis(gt_hypothesis, gene_hyp)
            
            if scores_dict is None:
                self.extraction_failures.increment()
            
            return {
                'file': cur_file,
                'step_idx': idx,
                'tier': tier,
                'scores': scores_dict,
                'total_score': total_score,
                'generated_hypothesis': gene_hyp,
                'reasoning_trace': reasoning_trace,
                'ground_truth_hypothesis': gt_hypothesis,
                'bounded_title': bounded_info['bounded_title'],
                'bounded_similarity': bounded_info['bounded_similarity'],
                'gt_inspiration_title': bounded_info['gt_inspiration_title'],
                'extraction_failed': scores_dict is None
            }
        
        if self.eval_max_workers > 1 and len(eval_tasks) > 1:
            with ThreadPoolExecutor(max_workers=self.eval_max_workers) as executor:
                results = list(executor.map(eval_task, eval_tasks))
            file_results = [r for r in results if r is not None]
        else:
            for task in eval_tasks:
                result = eval_task(task)
                if result:
                    file_results.append(result)
        
        return file_results

    def evaluate_dataset(self, output_path: str = None):
        """Evaluate the model on bounded selections dataset.
        
        Uses PIPELINED execution for maximum GPU utilization:
        - Generation and evaluation run in parallel
        - As soon as a batch is generated, it's queued for evaluation
        - Both generation GPU and evaluation API GPU stay busy
        """
        import queue
        import threading
        
        files = sorted([f for f in os.listdir(self.bounded_selections_dir) if f.endswith('.json')])
        
        if not files:
            print(f"Warning: No JSON files found in {self.bounded_selections_dir}")
            return []
        
        # Apply max_files limit if specified
        if self.max_files and self.max_files < len(files):
            files = files[:self.max_files]
            print(f"Limiting to {self.max_files} files (testing mode)")
        
        print(f"Found {len(files)} files to evaluate")
        print(f"Evaluating tiers: {self.tiers}")
        
        # ========== Phase 1: Collect all generation tasks ==========
        print("\n[Phase 1] Collecting generation tasks...")
        all_gen_tasks = []  # (file_name, idx, tier, prompt, gt_hypothesis, bounded_info)
        
        for cur_file in tqdm(files, desc="Reading files"):
            file_path = os.path.join(self.bounded_selections_dir, cur_file)
            if not os.path.exists(file_path):
                continue
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            research_question = data.get('research_question', '')
            background_survey = data.get('background_survey', '')
            inspirations = data.get('inspirations', [])
            
            if not research_question or not inspirations:
                continue
            
            prev_hypothesis = "No previous hypothesis."
            
            for insp in inspirations:
                idx = insp.get('idx', 0)
                gt_hypothesis = insp.get('delta_hypothesis', '')
                bounded_selections = insp.get('bounded_selections', {})
                gt_title = insp.get('gt_title', '')
                
                if not gt_hypothesis:
                    continue
                
                for tier in self.tiers:
                    bounded = bounded_selections.get(tier)
                    if bounded is None:
                        continue
                    
                    bounded_title = bounded.get('title', '')
                    bounded_abstract = bounded.get('abstract', '')
                    similarity = bounded.get('similarity', 0)
                    
                    if not bounded_title or not bounded_abstract:
                        continue
                    
                    prompt = self.build_prompt(
                        research_question,
                        background_survey,
                        prev_hypothesis,
                        bounded_title,
                        bounded_abstract
                    )
                    
                    bounded_info = {
                        'bounded_title': bounded_title,
                        'bounded_similarity': similarity,
                        'gt_inspiration_title': gt_title
                    }
                    
                    all_gen_tasks.append((cur_file, idx, tier, prompt, gt_hypothesis, bounded_info))
                
                if gt_hypothesis:
                    if prev_hypothesis == "No previous hypothesis.":
                        prev_hypothesis = gt_hypothesis
                    else:
                        prev_hypothesis = prev_hypothesis + "\n\n" + gt_hypothesis
        
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
                    
                    task_idx, task, response = item
                    file_name, idx, tier, prompt, gt_hypothesis, bounded_info = task
                    
                    self.total_evaluations.increment()
                    
                    # Extract hypothesis
                    gene_hyp = self.extract_hypothesis_from_response(response)
                    
                    # Extract reasoning
                    think_start = response.find('<think>')
                    think_end = response.find('</think>')
                    if think_end != -1:
                        start_pos = think_start + len('<think>') if think_start != -1 else 0
                        reasoning_trace = response[start_pos:think_end].strip()
                    else:
                        reasoning_trace = response
                    
                    # Score via API
                    scores_dict, total_score = self.score_single_hypothesis(gt_hypothesis, gene_hyp)
                    
                    if scores_dict is None:
                        self.extraction_failures.increment()
                    
                    result = {
                        'file': file_name,
                        'step_idx': idx,
                        'tier': tier,
                        'scores': scores_dict,
                        'total_score': total_score,
                        'generated_hypothesis': gene_hyp,
                        'reasoning_trace': reasoning_trace,
                        'ground_truth_hypothesis': gt_hypothesis,
                        'bounded_title': bounded_info['bounded_title'],
                        'bounded_similarity': bounded_info['bounded_similarity'],
                        'gt_inspiration_title': bounded_info['gt_inspiration_title'],
                        'extraction_failed': scores_dict is None
                    }
                    
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
        
        # Calculate metrics
        self._print_and_save_metrics(eval_results, output_path)
        
        return eval_results

    def _print_and_save_metrics(self, eval_results: List[Dict], output_path: str = None):
        """Calculate and print tier-wise metrics."""
        
        print("\n" + "="*70)
        print("BOUNDED COMPOSITION EVALUATION SUMMARY (Rubric Scoring)")
        print("="*70)
        
        # Group by tier
        tier_results = {tier: [] for tier in self.tiers}
        for r in eval_results:
            tier = r.get('tier')
            if tier in tier_results and not r.get('extraction_failed', False):
                if r.get('total_score') is not None:
                    tier_results[tier].append(r)
        
        # Calculate overall and per-tier metrics
        all_scores = []
        tier_metrics = {}
        
        for tier in self.tiers:
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
        
        # Print tier-wise results
        print("\n=== Per-Tier Results ===")
        print(f"{'Tier':<10} {'Count':>8} {'Mean Score':>12} {'Motivation':>12} {'Mechanism':>12} {'Methodology':>12}")
        print("-" * 70)
        
        for tier in self.tiers:
            metrics = tier_metrics[tier]
            if metrics:
                print(f"{tier:<10} {metrics['count']:>8} {metrics['mean_total_score']:>12.2f} "
                      f"{metrics['mean_motivation']:>12.2f} {metrics['mean_mechanism']:>12.2f} "
                      f"{metrics['mean_methodology']:>12.2f}")
            else:
                print(f"{tier:<10} {'N/A':>8} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
        
        print("-" * 70)
        
        # Overall statistics (across all tiers)
        all_valid_results = [r for r in eval_results if not r.get('extraction_failed', False) and r.get('scores')]
        
        if all_valid_results:
            overall_mean = sum(r['total_score'] for r in all_valid_results) / len(all_valid_results)
            overall_motivation = sum(r['scores']['motivation'] for r in all_valid_results) / len(all_valid_results)
            overall_mechanism = sum(r['scores']['mechanism'] for r in all_valid_results) / len(all_valid_results)
            overall_methodology = sum(r['scores']['methodology'] for r in all_valid_results) / len(all_valid_results)
            
            print(f"\n=== Overall Statistics (All Tiers Combined) ===")
            print(f"Mean Total Score: {overall_mean:.2f} / 12")
            print(f"Mean Motivation:  {overall_motivation:.2f} / 4")
            print(f"Mean Mechanism:   {overall_mechanism:.2f} / 4")
            print(f"Mean Methodology: {overall_methodology:.2f} / 4")
            print(f"Total evaluations: {len(all_valid_results)}")
            all_total_scores = [r['total_score'] for r in all_valid_results]
            print(f"Min: {min(all_total_scores):.2f}, Max: {max(all_total_scores):.2f}")
        
        # Extraction failures
        total_evals = self.total_evaluations.value
        extraction_fails = self.extraction_failures.value
        if extraction_fails > 0:
            print(f"\nExtraction failures: {extraction_fails}/{total_evals} ({extraction_fails/total_evals:.1%})")
        else:
            print(f"\n✓ All {total_evals} evaluations completed successfully.")
        
        # Save results
        if output_path:
            output_folder = output_path.rstrip('.json')
            os.makedirs(output_folder, exist_ok=True)
            
            # 1. metrics.json - successful evaluations
            metrics_data = []
            for r in eval_results:
                if not r.get('extraction_failed', False) and r.get('scores'):
                    metrics_data.append({
                        'file': r['file'],
                        'step_idx': r['step_idx'],
                        'tier': r['tier'],
                        'scores': r['scores'],
                        'total_score': r['total_score'],
                        'bounded_similarity': r.get('bounded_similarity')
                    })
            
            with open(os.path.join(output_folder, 'metrics.json'), 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            # 2. generations.json - all results
            generations_data = []
            for r in eval_results:
                generations_data.append({
                    'file': r['file'],
                    'step_idx': r['step_idx'],
                    'tier': r['tier'],
                    'scores': r.get('scores'),
                    'total_score': r.get('total_score'),
                    'generated_hypothesis': r.get('generated_hypothesis', ''),
                    'reasoning_trace': r.get('reasoning_trace', ''),
                    'ground_truth_hypothesis': r.get('ground_truth_hypothesis', ''),
                    'bounded_title': r.get('bounded_title', ''),
                    'bounded_similarity': r.get('bounded_similarity'),
                    'gt_inspiration_title': r.get('gt_inspiration_title', ''),
                    'extraction_failed': r.get('extraction_failed', False)
                })
            
            with open(os.path.join(output_folder, 'generations.json'), 'w') as f:
                json.dump(generations_data, f, indent=2)
            
            # 3. summary.json - tier-wise and overall statistics
            # Calculate overall dimension scores
            all_valid = [r for r in eval_results if not r.get('extraction_failed', False) and r.get('scores')]
            overall_metrics = None
            if all_valid:
                overall_metrics = {
                    'mean_total_score': sum(r['total_score'] for r in all_valid) / len(all_valid),
                    'mean_motivation': sum(r['scores']['motivation'] for r in all_valid) / len(all_valid),
                    'mean_mechanism': sum(r['scores']['mechanism'] for r in all_valid) / len(all_valid),
                    'mean_methodology': sum(r['scores']['methodology'] for r in all_valid) / len(all_valid),
                    'min_score': min(r['total_score'] for r in all_valid),
                    'max_score': max(r['total_score'] for r in all_valid)
                }
            
            summary_data = {
                'overall_metrics': overall_metrics,
                'tier_metrics': tier_metrics,
                'tiers_evaluated': self.tiers,
                'total_evaluations': len(all_valid) if all_valid else 0,
                'extraction_failures': extraction_fails
            }
            
            with open(os.path.join(output_folder, 'summary.json'), 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            print(f"\nResults saved to: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate bounded composition using rubric scoring')
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA checkpoint")
    
    # Evaluation settings
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--max_length", type=int, default=16384, help="Maximum sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Maximum new tokens")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    
    # API settings for LLM evaluation
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Eval model")
    parser.add_argument("--api_type", type=int, default=0, help="API type")
    parser.add_argument("--api_key", type=str, default="", help="API key")
    parser.add_argument("--base_url", type=str, default="", help="Base URL")
    parser.add_argument("--eval_max_workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--api_max_tokens", type=int, default=16384, help="API max tokens")
    
    # Dataset paths - BOUNDED MODE
    parser.add_argument("--bounded_selections_dir", type=str, required=True,
                       help="Path to bounded selections directory")
    parser.add_argument("--tiers", type=str, default="hard,medium,easy",
                       help="Comma-separated list of tiers to evaluate")
    
    # Output path
    parser.add_argument("--eval_result_path", type=str, required=True,
                       help="Path to save evaluation results")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum files to process (for testing)")
    
    args = parser.parse_args()
    
    # Parse tiers
    tiers = [t.strip() for t in args.tiers.split(',')]
    
    # Initialize evaluator
    evaluator = BoundedCompositionEvaluator(
        model_path=args.model_path,
        lora_path=args.lora_path,
        device="cuda",
        load_in_8bit=args.load_in_8bit,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        bounded_selections_dir=args.bounded_selections_dir,
        tiers=tiers,
        api_type=args.api_type,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        batch_size=args.batch_size,
        eval_max_workers=args.eval_max_workers,
        api_max_tokens=args.api_max_tokens,
        max_files=args.max_files
    )
    
    # Run evaluation
    evaluator.evaluate_dataset(output_path=args.eval_result_path)

