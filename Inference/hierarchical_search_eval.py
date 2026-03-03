#!/usr/bin/env python3
"""
Hierarchical Search Evaluation with IR Model

Tests hierarchical search using the trained IR model to find ground truth inspirations.
Uses SGLang for fast inference with logprobs extraction.
"""

import os
import sys
import json
import time
import math
import hashlib
from glob import glob
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Preprocessing.hierarchical_search.tree_search import HierarchicalSearchTree
from Inference.ir_probability_extractor import IRProbabilityExtractor, build_ir_prompt, LABELS
from Inference.eval_utils import parse_motivation_from_hypothesis_component, truncate_before_problem


@dataclass
class SearchResult:
    """Result of a single hierarchical search."""
    found: bool
    depth: int
    inference_calls: int
    path: List[str]  # node_ids
    sample_id: str  # Unique ID for this eval sample (for resume)
    gt_paper_id: str
    gt_title: str
    search_time: float
    propose_rank: int = 0  # For best-first: which proposal found gt (0 if not found)


class ProbabilityCache:
    """Thread-safe cache for visited node probabilities."""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, float]] = {}
        self.hits: int = 0
        self.misses: int = 0
        self._lock = threading.Lock()
    
    def get_key(self, node_id: str, research_question: str, background_survey: str, 
                previous_hypothesis: Optional[str] = None) -> str:
        """Generate cache key from node and context."""
        # Hash the context to create a unique key
        # Include previous_hypothesis to differentiate samples with different insp_idx
        prev_hyp_part = previous_hypothesis or ""
        context_hash = hashlib.md5(f"{research_question}|{background_survey}|{prev_hyp_part}".encode()).hexdigest()[:8]
        return f"{node_id}_{context_hash}"
    
    def get(self, key: str) -> Optional[Dict[str, float]]:
        with self._lock:
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def set(self, key: str, probs: Dict[str, float]):
        with self._lock:
            self.cache[key] = probs


class HierarchicalSearchEvaluator:
    """Evaluates hierarchical search with IR model via SGLang."""
    
    def __init__(
        self,
        tree_dir: str,
        sglang_urls: List[str] = None,
        use_cache: bool = True,
        softmax_temperature: float = 1.0,
        max_tokens: int = 20480,
        top_logprobs: int = 30
    ):
        """
        Args:
            tree_dir: Path to hierarchical search tree
            sglang_urls: List of SGLang endpoint URLs for load balancing
            use_cache: Cache inference results for same (node, query) pairs
            softmax_temperature: Temperature for probability scaling
            max_tokens: Max tokens for LLM output
            top_logprobs: Number of top logprobs to return (should cover all candidates)
        """
        self.use_cache = use_cache
        self.cache = ProbabilityCache()  # Thread-safe cache
        self.softmax_temperature = softmax_temperature
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self._inference_lock = threading.Lock()  # For thread-safe inference counting
        
        # Load tree
        print(f"Loading tree from {tree_dir}...")
        self.tree = HierarchicalSearchTree.load(tree_dir)
        stats = self.tree.get_stats()
        print(f"  Papers: {stats['num_papers']}, Depth: {stats['max_depth']}, Branching: {stats['branching_factor']}")
        
        # Build paper title index for finding ground truth
        self.title_to_paper_id = {}
        for pid, paper in self.tree.papers.items():
            self.title_to_paper_id[paper['title'].lower().strip()] = pid
        
        # Initialize SGLang extractor with load balancing
        print(f"Using SGLang API at {sglang_urls}")
        self.extractor = IRProbabilityExtractor(base_urls=sglang_urls)
        print("Ready for evaluation!")
    
    def infer_selection(
        self,
        research_question: str,
        background_survey: str,
        candidates: List[Dict],
        previous_hypothesis: Optional[str] = None,
        node_id: Optional[str] = None
    ) -> Tuple[Dict[str, float], int]:
        """
        Run IR model inference to get selection probabilities.
        
        Returns:
            (probabilities, selected_index)
        """
        num_candidates = len(candidates)
        valid_labels = LABELS[:num_candidates]
        
        # Check cache
        if self.use_cache and node_id:
            cache_key = self.cache.get_key(node_id, research_question, background_survey, previous_hypothesis)
            # cached: {"A": 0.45, "B": 0.30, "C": 0.15, ...}
            cached = self.cache.get(cache_key)
            if cached:
                selected = max(cached, key=cached.get)
                return cached, LABELS.index(selected)
        
        # Call SGLang
        result = self.extractor.get_selection_probabilities(
            research_question=research_question,
            background_survey=background_survey,
            candidates=candidates,
            previous_hypothesis=previous_hypothesis,
            softmax_temperature=self.softmax_temperature,
            max_tokens=self.max_tokens,
            top_logprobs=self.top_logprobs
        )
        probs = result.probabilities
        
        # Cache
        if self.use_cache and node_id:
            cache_key = self.cache.get_key(node_id, research_question, background_survey, previous_hypothesis)
            self.cache.set(cache_key, probs)
        
        return probs, result.selected_index
    
    def find_paper_in_subtree(self, node: Dict, target_paper_id: str) -> bool:
        """Check if target paper is in this subtree."""
        if node.get('is_leaf'):
            # Leaf node must have paper_id
            return node['paper_id'] == target_paper_id
        
        # Non-leaf node must have children
        for child in node['children']:
            if self.find_paper_in_subtree(child, target_paper_id):
                return True
        return False
    
    def search_greedy(
        self,
        research_question: str,
        background_survey: str,
        sample_id: str,
        gt_paper_id: str,
        gt_title: str,
        previous_hypothesis: Optional[str] = None,
        max_depth: int = 10,
        verbose: bool = False
    ) -> SearchResult:
        """Perform greedy hierarchical search to find a specific paper."""
        start_time = time.time()
        
        current = self.tree.root
        path = [current['node_id']]
        inference_calls = 0
        
        while not current.get('is_leaf') and len(path) <= max_depth:
            candidates = self.tree.get_candidates_at_node(current)
            
            if len(candidates) <= 1:
                # No choice, auto-navigate
                if candidates:
                    current = self.tree.get_child_by_index(current, 0)
                    path.append(current['node_id'])
                break
            
            # Convert to IR format
            ir_candidates = [{"title": c.title, "abstract": c.abstract} for c in candidates]
            
            # Get probabilities
            probs, selected_idx = self.infer_selection(
                research_question=research_question,
                background_survey=background_survey,
                candidates=ir_candidates,
                previous_hypothesis=previous_hypothesis,
                node_id=current['node_id']
            )
            inference_calls += 1
            
            # Find which branch contains ground truth (for analysis)
            gt_branch_idx = -1
            # current is not a leaf here, must have children
            for i, child in enumerate(current['children']):
                if self.find_paper_in_subtree(child, gt_paper_id):
                    gt_branch_idx = i
                    break
            
            if verbose:
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                top3 = sorted_probs[:3]
                gt_label = LABELS[gt_branch_idx] if gt_branch_idx >= 0 else "?"
                gt_prob = probs.get(gt_label, 0)
                print(f"    Level {current['level']}: selected [{LABELS[selected_idx]}] "
                      f"(gt=[{gt_label}] p={gt_prob:.3f}), top3={[(l, f'{p:.3f}') for l,p in top3]}")
            
            # Navigate to selected child
            current = self.tree.get_child_by_index(current, selected_idx)
            if current is None:
                break
            path.append(current['node_id'])
            
            # Auto-skip single-child nodes
            while not current.get('is_leaf') and len(current['children']) == 1:
                current = current['children'][0]
                path.append(current['node_id'])
        
        # Check if found (if leaf, must have paper_id)
        found = current.get('is_leaf') and current['paper_id'] == gt_paper_id
        
        return SearchResult(
            found=found,
            depth=len(path),
            inference_calls=inference_calls,
            path=path,
            sample_id=sample_id,
            gt_paper_id=gt_paper_id,
            gt_title=gt_title,
            search_time=time.time() - start_time
        )
    
    def search_beam(
        self,
        research_question: str,
        background_survey: str,
        sample_id: str,
        gt_paper_id: str,
        gt_title: str,
        previous_hypothesis: Optional[str] = None,
        beam_width: int = 3,
        max_depth: int = 10
    ) -> SearchResult:
        """Perform beam search to find a specific paper."""
        start_time = time.time()
        inference_calls = 0
        
        # Each beam item: (node, path, cumulative_log_prob)
        beams = [(self.tree.root, [self.tree.root['node_id']], 0.0)]
        
        while beams:
            new_beams = []
            
            for current, path, cum_log_prob in beams:
                if current.get('is_leaf'):
                    # Check if found (leaf must have paper_id)
                    if current['paper_id'] == gt_paper_id:
                        return SearchResult(
                            found=True,
                            depth=len(path),
                            inference_calls=inference_calls,
                            path=path,
                            sample_id=sample_id,
                            gt_paper_id=gt_paper_id,
                            gt_title=gt_title,
                            search_time=time.time() - start_time
                        )
                    continue
                
                if len(path) > max_depth:
                    continue
                
                candidates = self.tree.get_candidates_at_node(current)
                
                if len(candidates) <= 1:
                    if candidates:
                        child = self.tree.get_child_by_index(current, 0)
                        new_beams.append((child, path + [child['node_id']], cum_log_prob))
                    continue
                
                # Get probabilities
                ir_candidates = [{"title": c.title, "abstract": c.abstract} for c in candidates]
                probs, _ = self.infer_selection(
                    research_question=research_question,
                    background_survey=background_survey,
                    candidates=ir_candidates,
                    previous_hypothesis=previous_hypothesis,
                    node_id=current['node_id']
                )
                inference_calls += 1
                
                # Expand top-k branches
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                for label, prob in sorted_probs[:beam_width]:
                    idx = LABELS.index(label)
                    child = self.tree.get_child_by_index(current, idx)
                    if child:
                        new_path = path + [child['node_id']]
                        new_log_prob = cum_log_prob + math.log(prob + 1e-10)
                        new_beams.append((child, new_path, new_log_prob))
            
            # Keep top beams by probability
            beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width * 2]
        
        return SearchResult(
            found=False,
            depth=0,
            inference_calls=inference_calls,
            path=[],
            sample_id=sample_id,
            gt_paper_id=gt_paper_id,
            gt_title=gt_title,
            search_time=time.time() - start_time
        )
    
    def search_best_first(
        self,
        research_question: str,
        background_survey: str,
        sample_id: str,
        gt_paper_id: str,
        gt_title: str,
        previous_hypothesis: Optional[str] = None,
        max_proposals: int = 100
    ) -> SearchResult:
        """
        Best-First Search using geometric mean probability.
        
        At each step, expand the node with highest geometric mean probability:
        score = prob^(1/depth) = exp(log_prob / depth)
        
        This allows fair comparison across different depths since deeper nodes
        naturally have smaller cumulative probabilities.
        
        Args:
            max_proposals: Maximum leaf nodes to propose before giving up
            
        Returns:
            SearchResult with:
            - inference_calls: number of IR model inferences needed
            - propose_rank: at which proposal the ground truth was found (1-indexed)
        """
        import heapq
        
        start_time = time.time()
        inference_calls = 0
        propose_count = 0
        
        # Priority queue element format (6 elements):
        #   (-score, depth, node_id, node, path, cum_log_prob)
        #   
        #   -score:       Negative geometric mean score for min-heap (heapq is min-heap)
        #                 score = cum_log_prob / depth = log(p1*p2*...*pn) / n
        #                 Higher score = higher priority, so we negate it
        #   depth:        Current depth in tree (for tie-breaking)
        #   node_id:      Python id(node) for deduplication
        #   node:         The actual tree node object
        #   path:         List of node_ids from root to current node
        #   cum_log_prob: Cumulative log probability = log(p1) + log(p2) + ... + log(pn)
        frontier = []
        
        # Initialize with root node
        # Root has no probability yet (cum_log_prob=0), so score=0
        root = self.tree.root
        heapq.heappush(frontier, (0.0, 0, id(root), root, [root['node_id']], 0.0))
        
        # Track expanded nodes to avoid re-expansion
        # We use id(node) (Python object identity) as the key because:
        # 1. Same node can be reached via different paths
        # 2. First time we pop a node, it's via the optimal path (highest score)
        # 3. Any subsequent pops of the same node are suboptimal, so we skip them
        expanded = set()
        
        # Main loop: always expand the highest-scoring unexpanded node
        # Stop when: (1) found GT, (2) proposed max_proposals leaves (if limit set), or (3) frontier empty
        while frontier and (max_proposals == 0 or propose_count < max_proposals):
            # Pop the node with highest geometric mean probability (lowest -score)
            neg_score, depth, node_id, current, path, cum_log_prob = heapq.heappop(frontier)
            
            # Skip if already expanded (can happen when same node added multiple times)
            if node_id in expanded:
                continue
            expanded.add(node_id)
            
            # Case 1: Leaf node - this is a paper, propose it as a candidate
            if current.get('is_leaf'):
                propose_count += 1
                # Check if this is the ground truth inspiration (leaf must have paper_id)
                if current['paper_id'] == gt_paper_id:
                    return SearchResult(
                        found=True,
                        depth=len(path),
                        inference_calls=inference_calls,
                        path=path,
                        sample_id=sample_id,
                        gt_paper_id=gt_paper_id,
                        gt_title=gt_title,
                        search_time=time.time() - start_time,
                        propose_rank=propose_count  # Key metric: which proposal found GT
                    )
                continue  # Not GT, continue searching other nodes
            
            # Case 2: Internal node - need to expand its children
            candidates = self.tree.get_candidates_at_node(current)
            
            # Case 2a: Single child (or no child) - no IR inference needed
            # When there's only one option, no need to ask IR model for selection
            if len(candidates) <= 1:
                if candidates:
                    child = self.tree.get_child_by_index(current, 0)
                    if child and id(child) not in expanded:
                        new_path = path + [child['node_id']]
                        new_depth = depth + 1
                        # No new probability added (no selection choice was made)
                        # cum_log_prob stays the same, but depth increases
                        # Note: Since cum_log_prob < 0 (log of prob < 1), 
                        #       dividing by larger depth makes score HIGHER (closer to 0)
                        #       So this child will have higher priority than its parent
                        new_score = cum_log_prob / new_depth if new_depth > 0 else 0.0
                        heapq.heappush(frontier, (-new_score, new_depth, id(child), child, new_path, cum_log_prob))
                continue  # Move to next frontier node
            
            # Case 2b: Multiple children - need IR model to get selection probabilities
            ir_candidates = [{"title": c.title, "abstract": c.abstract} for c in candidates]
            probs, _ = self.infer_selection(
                research_question=research_question,
                background_survey=background_survey,
                candidates=ir_candidates,
                previous_hypothesis=previous_hypothesis,
                node_id=current['node_id']
            )
            inference_calls += 1  # Key metric: how many IR inferences needed
            
            # Add ALL children to frontier (unlike greedy which only adds top-1)
            # Each child's priority is based on geometric mean probability
            #
            # probs format: {'A': 0.65, 'B': 0.20, 'C': 0.08, ...} 
            #   - Keys are labels A-O (always 15 labels from IR model's softmax)
            #   - Values are probabilities summing to 1.0
            #   - Ordered by label (A, B, C, ...), NOT by probability value
            for label, prob in probs.items():
                # Map label to candidate index: 'A'->0, 'B'->1, etc.
                idx = LABELS.index(label)
                # Boundary check: IR model outputs 15 probs, but we may have fewer candidates
                # e.g., if only 5 candidates, labels F-O (idx 5-14) are out of range
                if idx < len(candidates):
                    child = self.tree.get_child_by_index(current, idx)
                    if child and id(child) not in expanded:
                        new_path = path + [child['node_id']]
                        # Update cumulative log probability: log(p1*p2*...*pn) = sum(log(pi))
                        # Add 1e-10 to avoid log(0) when prob is very small
                        new_log_prob = cum_log_prob + math.log(prob + 1e-10)
                        new_depth = depth + 1
                        # Geometric mean score: (p1*p2*...*pn)^(1/n) = exp(log(p)/n)
                        # In log space: log(geometric_mean) = cum_log_prob / depth
                        # Higher score = better candidate
                        new_score = new_log_prob / new_depth
                        # heapq is a min-heap, so we negate score to get max-heap behavior
                        # (highest score = lowest -score = popped first)
                        heapq.heappush(frontier, (-new_score, new_depth, id(child), child, new_path, new_log_prob))
        
        return SearchResult(
            found=False,
            depth=0,
            inference_calls=inference_calls,
            path=[],
            sample_id=sample_id,
            gt_paper_id=gt_paper_id,
            gt_title=gt_title,
            search_time=time.time() - start_time,
            propose_rank=0
        )
    
    def _process_single_sample(
        self,
        sample: Dict,
        sample_idx: int,
        total_samples: int,
        search_mode: str,  # "greedy", "beam", "best_first"
        beam_width: int,
        max_proposals: int,
        verbose: bool
    ) -> SearchResult:
        """Process a single sample (for parallel execution)."""
        if search_mode == "beam":
            result = self.search_beam(
                research_question=sample['research_question'],
                background_survey=sample['background_survey'],
                sample_id=sample['sample_id'],
                gt_paper_id=sample['gt_paper_id'],
                gt_title=sample['gt_title'],
                previous_hypothesis=sample['prev_hyp'],
                beam_width=beam_width
            )
        elif search_mode == "best_first":
            result = self.search_best_first(
                research_question=sample['research_question'],
                background_survey=sample['background_survey'],
                sample_id=sample['sample_id'],
                gt_paper_id=sample['gt_paper_id'],
                gt_title=sample['gt_title'],
                previous_hypothesis=sample['prev_hyp'],
                max_proposals=max_proposals
            )
        else:  # greedy
            result = self.search_greedy(
                research_question=sample['research_question'],
                background_survey=sample['background_survey'],
                sample_id=sample['sample_id'],
                gt_paper_id=sample['gt_paper_id'],
                gt_title=sample['gt_title'],
                previous_hypothesis=sample['prev_hyp'],
                verbose=False  # Disable per-level verbose in parallel mode
            )
        
        if verbose:
            status = "✓ FOUND" if result.found else "✗ NOT FOUND"
            # For best_first: show propose_rank (which proposal found GT)
            rank_str = f", propose={result.propose_rank}" if result.propose_rank > 0 else ""
            print(f"[{sample_idx+1}/{total_samples}] {status} | "
                  f"depth={result.depth}, infer_calls={result.inference_calls}{rank_str}, "
                  f"time={result.search_time:.1f}s | {sample['gt_title'][:40]}...")
        
        return result
    
    def evaluate(
        self,
        eval_samples: List[Dict],
        incremental_output: str,  # Required: path to save results incrementally
        verbose: bool = True,
        search_mode: str = "greedy",  # "greedy", "beam", "best_first"
        beam_width: int = 3,
        max_proposals: int = 100,
        num_workers: int = 1
    ) -> Dict:
        """
        Evaluate on multiple samples in parallel.
        
        Args:
            search_mode: "greedy", "beam", or "best_first"
            beam_width: Width for beam search
            max_proposals: Max proposals for best-first search
            num_workers: Number of parallel workers (default 1).
            incremental_output: If set, save each result to this file as it completes (JSONL format).
                               This allows recovery if the process is interrupted.
        """
        results = []
        total = len(eval_samples)
        
        # Setup incremental output file (incremental_output is required)
        os.makedirs(os.path.dirname(incremental_output) or '.', exist_ok=True)
        incremental_file = open(incremental_output, 'a')  # Append mode
        print(f"Incremental results will be saved to: {incremental_output}")
        
        print(f"\nEvaluating {total} samples with {num_workers} workers...")
        try:
            with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
                futures = {
                    executor.submit(
                        self._process_single_sample,
                        sample, i, total, search_mode, beam_width, max_proposals, verbose
                    ): i for i, sample in enumerate(eval_samples)
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    
                    # Save incrementally
                    incremental_file.write(json.dumps(asdict(result)) + '\n')
                    incremental_file.flush()  # Ensure it's written to disk
        finally:
            incremental_file.close()
        
        # Sort by original order
        results.sort(key=lambda r: next(
            i for i, s in enumerate(eval_samples) if s['sample_id'] == r.sample_id
        ))
        
        # Compute stats
        found_count = sum(1 for r in results if r.found)
        total_infer = sum(r.inference_calls for r in results)
        total_time = sum(r.search_time for r in results)
        avg_depth = sum(r.depth for r in results) / len(results) if results else 0
        
        # For best-first: compute propose rank stats
        found_ranks = [r.propose_rank for r in results if r.found and r.propose_rank > 0]
        avg_propose_rank = sum(found_ranks) / len(found_ranks) if found_ranks else 0
        
        # Convert results to serializable format for saving
        results_dicts = [asdict(r) for r in results]
        
        return {
            'num_samples': len(results),
            'found': found_count,
            'accuracy': found_count / len(results) if results else 0,
            'avg_depth': avg_depth,
            'avg_inference_calls': total_infer / len(results) if results else 0,
            'total_inference_calls': total_infer,
            'avg_propose_rank': avg_propose_rank,  # For best-first
            'cache_hits': self.cache.hits,
            'cache_misses': self.cache.misses,
            'total_time': total_time,
            'avg_time_per_sample': total_time / len(results) if results else 0,
            'wall_time': max(r.search_time for r in results) if results else 0,  # Parallel wall time
            'results': results_dicts  # Detailed per-sample results
        }


def load_eval_samples(
    eval_dir: str, 
    tree: HierarchicalSearchTree, 
    max_samples: int = 0,
    motivation_option: int = 0,
    truncate_survey: int = 0
) -> List[Dict]:
    """
    Load evaluation samples from SFT QA data.
    
    Args:
        eval_dir: Directory containing eval JSON files
        tree: HierarchicalSearchTree for looking up paper IDs
        max_samples: Maximum samples to load (0 = all samples)
        motivation_option: 0=no motivation, 
                          1=simple motivation (from inspiration['motivation']),
                          2=detailed motivation (parsed from hypothesis_components).
                          This is for ablation study of Section 3.5 (Motivation Planning).
                          By providing oracle motivation, we can verify that good motivation
                          effectively guides hierarchical search.
        truncate_survey: 0=no truncation, 1=truncate background_survey before problem/limitation
                        keywords (e.g., "however", "limitation", "gap"). This removes implicit
                        motivation from background, allowing proper evaluation of explicit motivation.
    
    Returns:
        List of sample dicts with research_question, background_survey, prev_hyp, gt_title, gt_paper_id
    """
    # Build title index
    title_to_pid = {}
    for pid, paper in tree.papers.items():
        title_to_pid[paper['title'].lower().strip()] = pid
    
    samples = []
    skipped = 0
    truncated_count = 0
    eval_files = sorted(glob(os.path.join(eval_dir, "*.json")))
    
    for ef in eval_files:
        with open(ef) as f:
            data = json.load(f)
        
        # hypothesis_components: {"0": delta_hyp_0, "1": delta_hyp_1, ...}
        # Each key corresponds to an inspiration index
        hyp_components = data['hypothesis_components']  # Must exist
        
        for insp_idx, insp in enumerate(data['inspiration']):  # Must exist
            gt_title = insp['found_title'].lower().strip()  # Must exist
            if gt_title not in title_to_pid:
                # Paper not in corpus - skip (data mismatch, not a bug)
                skipped += 1
                continue
            
            # Build prev_hyp as cumulative (join all previous deltas)
            # Following the pattern from hypothesis_composition_sampling.py
            if insp_idx > 0:
                prev_deltas = [hyp_components[str(j)] for j in range(insp_idx)]  # Must exist
                prev_hyp = "\n\n".join(prev_deltas)
            else:
                prev_hyp = None  # No previous hypothesis for first inspiration
            
            # Unique sample_id = filename (without .json) + inspiration index
            file_base = os.path.basename(ef).replace('.json', '')
            sample_id = f"{file_base}_insp{insp_idx}"
            
            # Build background_survey, optionally with motivation hint
            background_survey = data['background_survey']
            
            # Optionally truncate background before problem descriptions
            # This removes implicit motivation (e.g., "However, limitation X...")
            # so that explicit motivation injection can be properly evaluated
            if truncate_survey:
                background_survey, was_truncated, _ = truncate_before_problem(background_survey)
                if was_truncated:
                    truncated_count += 1
            
            if motivation_option == 1:
                # Simple motivation from inspiration['motivation']
                # Format: Problem/Gap + Solution Direction + Why
                gt_motivation = insp.get('motivation', '')
                if gt_motivation:
                    # Append without strong directive label to avoid biasing model
                    background_survey = (
                        f"{background_survey}\n\n"
                        f"{gt_motivation}"
                    )
            elif motivation_option == 2:
                # Detailed motivation parsed from hypothesis_components
                # Format: Full "- Motivation (WHY): ..." section
                hyp_comp = hyp_components.get(str(insp_idx), '')
                detailed_motivation = parse_motivation_from_hypothesis_component(hyp_comp)
                if detailed_motivation:
                    # Append without strong directive label
                    background_survey = (
                        f"{background_survey}\n\n"
                        f"{detailed_motivation}"
                    )
            
            samples.append({
                'sample_id': sample_id,  # Unique ID for resume
                'file': os.path.basename(ef),
                'research_question': data['research_question'],
                'background_survey': background_survey,
                'prev_hyp': prev_hyp,
                'gt_title': insp['found_title'],
                'gt_paper_id': title_to_pid[gt_title]
            })
        
        # Stop early if we have enough samples (0 = unlimited)
        if max_samples > 0 and len(samples) >= max_samples:
            break
    
    if skipped > 0:
        print(f"  Warning: skipped {skipped} samples (paper not in corpus)")
    if truncate_survey:
        print(f"  Survey truncation: {truncated_count}/{len(samples)} samples truncated")
    return samples[:max_samples] if max_samples > 0 else samples


def compute_and_print_summary(results_list: List[Dict]) -> Dict:
    """Compute summary statistics from results and print to stdout."""
    if not results_list:
        summary = {
            'num_samples': 0, 'found': 0, 'accuracy': 0, 'avg_depth': 0,
            'avg_inference_calls': 0, 'total_inference_calls': 0,
            'avg_propose_rank': 0, 'total_time': 0, 'avg_time_per_sample': 0
        }
    else:
        found_count = sum(1 for r in results_list if r['found'])
        total_infer = sum(r['inference_calls'] for r in results_list)
        total_time = sum(r['search_time'] for r in results_list)
        avg_depth = sum(r['depth'] for r in results_list) / len(results_list)
        found_ranks = [r['propose_rank'] for r in results_list if r['found'] and r['propose_rank'] > 0]
        avg_propose_rank = sum(found_ranks) / len(found_ranks) if found_ranks else 0
        
        summary = {
            'num_samples': len(results_list),
            'found': found_count,
            'accuracy': found_count / len(results_list),
            'avg_depth': avg_depth,
            'avg_inference_calls': total_infer / len(results_list),
            'total_inference_calls': total_infer,
            'avg_propose_rank': avg_propose_rank,
            'total_time': total_time,
            'avg_time_per_sample': total_time / len(results_list),
        }
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    
    return summary


def save_results(summary: Dict, results_list: List[Dict], output_dir: str):
    """Save summary and results to files."""
    # Save results
    results_path = os.path.join(output_dir, "results.jsonl")
    with open(results_path, 'w') as f:
        for r in results_list:
            f.write(json.dumps(r) + '\n')
    print(f"\nResults saved to: {results_path}")
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hierarchical Search Evaluation")
    parser.add_argument("--tree-dir",
                       default="${SEARCH_TREE_DIR}")
    parser.add_argument("--eval-dir",
                       default="${TOMATO_STAR_EVAL_DIR}")
    parser.add_argument("--sglang-urls", nargs="+", 
                       default=["http://127.0.0.1:30000/v1"],
                       help="SGLang endpoint URLs (multiple for load balancing)")
    parser.add_argument("--max-samples", type=int, default=0,
                       help="Max samples to evaluate (0 = all samples)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable probability caching (cache saves inference results for same node+query, "
                            "useful when same research question visits same tree node multiple times)")
    parser.add_argument("--search-mode", type=str, default="greedy",
                       choices=["greedy", "beam", "best_first"],
                       help="Search strategy: greedy (default), beam, or best_first")
    parser.add_argument("--beam-width", type=int, default=3,
                       help="[beam] Number of top branches to expand at EACH hierarchy level")
    parser.add_argument("--max-proposals", type=int, default=0,
                       help="[best_first] Maximum leaf nodes to propose before giving up (0 = no limit, search until found)")
    parser.add_argument("--softmax-temperature", type=float, default=1.0, 
                       help="Softmax temperature for probability scaling (>1 = flatter distribution)")
    parser.add_argument("--max-tokens", type=int, default=20480,
                       help="Max tokens for LLM output (default 20k, fit within 32k context)")
    parser.add_argument("--top-logprobs", type=int, default=30,
                       help="Number of top logprobs to return per token (should cover all candidates A-O)")
    parser.add_argument("--num-workers", type=int, default=1,
                       help="Number of parallel workers (1=sequential, >1=parallel samples)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save results (required for incremental save and resume)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous run. Skips samples that are already in results_incremental.jsonl")
    parser.add_argument("--motivation-option", type=int, default=0, choices=[0, 1, 2],
                       help="[Ablation for Section 3.5] Motivation hint option: "
                            "0=no motivation (baseline), "
                            "1=simple motivation (from inspiration['motivation']: Problem/Gap + Solution Direction), "
                            "2=detailed motivation (parsed from hypothesis_components: full Motivation (WHY) section). "
                            "Simulates a perfect Motivation Planner to verify that good motivation "
                            "effectively guides hierarchical search (semantic teleportation).")
    parser.add_argument("--truncate-survey", type=int, default=0, choices=[0, 1],
                       help="[Ablation] Truncate background_survey before problem/limitation keywords. "
                            "0=no truncation (default), 1=truncate before 'however'/'limitation'/'gap'/etc. "
                            "This removes implicit motivation from the background, so explicit motivation "
                            "(--motivation-option 1 or 2) can be properly evaluated.")
    args = parser.parse_args()
    
    # Determine mode string for display
    if args.search_mode == "beam":
        mode_str = f"Beam Search (width={args.beam_width})"
    elif args.search_mode == "best_first":
        mp_str = "unlimited" if args.max_proposals == 0 else str(args.max_proposals)
        mode_str = f"Best-First Search (max_proposals={mp_str})"
    elif args.search_mode == "greedy":
        mode_str = "Greedy"
    else:
        raise ValueError(f"Invalid search mode: {args.search_mode}")
    
    print("="*60)
    print("Hierarchical Search Evaluation")
    print("="*60)
    print(f"Mode: {mode_str}")
    print(f"Softmax Temperature: {args.softmax_temperature}")
    print(f"Max Tokens: {args.max_tokens}, Top Logprobs: {args.top_logprobs}")
    print(f"SGLang Endpoints: {len(args.sglang_urls)}x")
    print(f"Parallel Workers: {args.num_workers}")
    print(f"Max Samples: {'all' if args.max_samples == 0 else args.max_samples}")
    if args.truncate_survey:
        print(f"*** Survey Truncation: ENABLED (removing implicit motivation from background) ***")
    if args.motivation_option == 1:
        print(f"*** Motivation Option: SIMPLE (Section 3.5 ablation - Problem/Gap + Solution Direction) ***")
    elif args.motivation_option == 2:
        print(f"*** Motivation Option: DETAILED (Section 3.5 ablation - full Motivation (WHY) section) ***")
    
    # Initialize evaluator
    evaluator = HierarchicalSearchEvaluator(
        tree_dir=args.tree_dir,
        sglang_urls=args.sglang_urls,
        use_cache=not args.no_cache,
        softmax_temperature=args.softmax_temperature,
        max_tokens=args.max_tokens,
        top_logprobs=args.top_logprobs
    )
    
    # Load samples
    print(f"\nLoading eval samples from {args.eval_dir}...")
    samples = load_eval_samples(
        args.eval_dir, 
        evaluator.tree, 
        max_samples=args.max_samples,
        motivation_option=args.motivation_option,
        truncate_survey=args.truncate_survey
    )
    print(f"Loaded {len(samples)} samples")
    
    # Setup incremental save path and handle resume
    os.makedirs(args.output_dir, exist_ok=True)
    incremental_path = os.path.join(args.output_dir, "results_incremental.jsonl")
    previous_results = []
    
    if os.path.exists(incremental_path):
        if not args.resume:
            # Safety check: abort to prevent data loss
            raise RuntimeError(
                f"\n{'='*60}\n"
                f"ERROR: Existing results found at {incremental_path}\n"
                f"{'='*60}\n"
                f"Options:\n"
                f"  1. Add --resume to continue from previous run\n"
                f"  2. Delete the file manually to start fresh\n"
                f"  3. Use a different --output-dir\n"
                f"{'='*60}"
            )
        
        # Resume: load previous results and filter completed samples
        completed_ids = set()
        with open(incremental_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        result = json.loads(line)
                        completed_ids.add(result['sample_id'])
                        previous_results.append(result)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed line: {e}")
        
        original_count = len(samples)
        samples = [s for s in samples if s['sample_id'] not in completed_ids]
        print(f"Resume mode: loaded {len(previous_results)} previous results, {len(samples)} remaining")
        
        if len(samples) == 0:
            print("All samples already completed! Generating final summary...")
            summary = compute_and_print_summary(previous_results)
            save_results(summary, previous_results, args.output_dir)
            sys.exit(0)
    
    # Evaluate
    print("\n" + "="*60)
    print("Running hierarchical search...")
    print("="*60)
    
    summary = evaluator.evaluate(
        samples,
        incremental_output=incremental_path,
        verbose=True,
        search_mode=args.search_mode,
        beam_width=args.beam_width,
        max_proposals=args.max_proposals,
        num_workers=args.num_workers
    )
    
    # Merge with previous results if resuming
    results_list = summary.pop('results', [])  # Current run results
    cache_hits = summary.get('cache_hits', 0)
    cache_misses = summary.get('cache_misses', 0)
    
    if previous_results:
        all_results = previous_results + results_list
        print(f"\nMerging {len(previous_results)} previous + {len(results_list)} new = {len(all_results)} total results")
        results_list = all_results
    
    # Compute final summary
    summary = compute_and_print_summary(results_list)
    summary['cache_hits'] = cache_hits
    summary['cache_misses'] = cache_misses
    
    save_results(summary, results_list, args.output_dir)
