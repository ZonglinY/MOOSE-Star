#!/usr/bin/env python3
"""
Tournament Search Evaluation (Baseline for Hierarchical Search)

Bottom-up tournament-style search that visits ALL leaf nodes, unlike hierarchical 
search which only follows one path from root to leaf.

Complexity:
- Hierarchical Search: O(log N) inference calls
- Tournament Search: O(N/K) inference calls (~218 for N=3035, K=15)
"""

import os
import sys
import json
import time
import math
from glob import glob
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Preprocessing.hierarchical_search.tree_search import HierarchicalSearchTree
from Inference.ir_probability_extractor import IRProbabilityExtractor
from Inference.eval_utils import parse_motivation_from_hypothesis_component, truncate_before_problem


@dataclass
class TournamentSearchResult:
    """Result of tournament search (compatible with hierarchical search output)."""
    found: bool
    depth: int                    # How many levels GT survived
    inference_calls_before_break: int  # IR calls until GT eliminated or won
    path: List[str]               # Empty (no path concept in tournament)
    sample_id: str
    gt_paper_id: str
    gt_title: str
    search_time: float
    propose_rank: float           # Estimated rank: winners_above + avg_rank_in_level


class TournamentSearchEvaluator:
    """
    Bottom-up tournament search through all papers.
    
    Level 1: Group leaves by tree parent → select winners
    Level 2+: Chunk winners into groups of K → select winners
    Repeat until 1 winner remains
    """
    
    def __init__(
        self,
        tree_dir: str,
        sglang_urls: List[str],
        softmax_temperature: float = 1.0,
        max_tokens: int = 20480,
        top_logprobs: int = 30,
        branching_factor: int = 15
    ):
        self.softmax_temperature = softmax_temperature
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.K = branching_factor
        
        # Load tree
        print(f"Loading tree from {tree_dir}...")
        self.tree = HierarchicalSearchTree.load(tree_dir)
        stats = self.tree.get_stats()
        self.num_papers = stats['num_papers']
        print(f"  Papers: {self.num_papers}, Branching: {stats['branching_factor']}")
        
        # Compute tournament structure
        # Level 1 groups by tree parent, Level 2+ chunks by K
        leaves = self._collect_leaves(self.tree.root)
        level1_groups = self._group_by_parent(leaves)
        level1_num_groups = len(level1_groups)
        level1_inference_groups = sum(1 for g in level1_groups if len(g) > 1)
        
        # Level sizes: [num_papers, num_level1_winners, num_level2_winners, ..., 1]
        self.level_sizes = [self.num_papers]
        n = level1_num_groups  # Level 1 produces this many winners
        self.level_sizes.append(n)
        while n > 1:
            n = math.ceil(n / self.K)
            self.level_sizes.append(n)
        
        # Expected calls: Level 1 (tree-based) + Level 2+ (K-based)
        level2plus_calls = sum(math.ceil(n / self.K) for n in self.level_sizes[1:-1])
        self.expected_calls = level1_inference_groups + level2plus_calls
        print(f"  Tournament: {self.level_sizes} → {self.expected_calls} calls/sample (L1={level1_inference_groups}, L2+={level2plus_calls})")
        
        # Initialize SGLang
        print(f"Using SGLang API at {sglang_urls}")
        self.extractor = IRProbabilityExtractor(base_urls=sglang_urls)
        print("Ready!")
    
    def _collect_leaves(self, node: Dict, parent_id: str = "root") -> List[Dict]:
        """Collect all leaves with parent info."""
        if node.get('is_leaf'):
            # Leaf node must have paper_id
            return [{'paper_id': node['paper_id'], 'parent_id': parent_id}]
        
        leaves = []
        node_id = node['node_id']
        # Non-leaf node must have children
        for child in node['children']:
            leaves.extend(self._collect_leaves(child, node_id))
        return leaves
    
    def _group_by_parent(self, items: List[Dict]) -> List[List[Dict]]:
        """Group by parent_id (for level 1 only).
        
        Returns: [group1, group2, ...]
            where each group = [{paper_id, parent_id}, {paper_id, parent_id}, ...]
        """
        groups = {}
        for item in items:
            # Every item must have parent_id from _collect_leaves
            groups.setdefault(item['parent_id'], []).append(item)
        return list(groups.values())
    
    def _chunk(self, items: List, size: int) -> List[List]:
        """Split into chunks of fixed size (for level 2+).
        
        Returns: same format as _group_by_parent
        """
        return [items[i:i + size] for i in range(0, len(items), size)]
    
    def _select_winner(
        self,
        candidates: List[Dict],
        research_question: str,
        background_survey: str,
        previous_hypothesis: Optional[str]
    ) -> int:
        """Run IR inference, return winner index."""
        ir_candidates = []
        for c in candidates:
            # paper_id must exist in tree.papers
            paper = self.tree.papers[c['paper_id']]
            ir_candidates.append({
                "title": paper['title'],
                "abstract": paper['abstract']
            })
        
        result = self.extractor.get_selection_probabilities(
            research_question=research_question,
            background_survey=background_survey,
            candidates=ir_candidates,
            previous_hypothesis=previous_hypothesis,
            softmax_temperature=self.softmax_temperature,
            max_tokens=self.max_tokens,
            top_logprobs=self.top_logprobs
        )
        return min(result.selected_index, len(candidates) - 1)
    
    def search(
        self,
        research_question: str,
        background_survey: str,
        sample_id: str,
        gt_paper_id: str,
        gt_title: str,
        previous_hypothesis: Optional[str] = None
    ) -> TournamentSearchResult:
        """Run tournament search."""
        start_time = time.time()
        inference_calls_before_break = 0
        
        # Collect all leaves
        candidates = self._collect_leaves(self.tree.root)
        
        # Track GT elimination
        gt_alive = any(c['paper_id'] == gt_paper_id for c in candidates)
        if not gt_alive:
            raise ValueError(f"GT paper {gt_paper_id} not found in candidates")
        gt_eliminated_level = 0  # Will be set to >0 when GT is eliminated
        
        level = 0
        while len(candidates) > 1:
            level += 1
            
            # Group candidates into matches
            # Level 1: group by tree parent (preserves tree structure)
            # Level 2+: chunk into groups of K (generic tournament bracket)
            # Both return [group1, group2, ...] where group = [{paper_id, parent_id}, ...]
            groups = self._group_by_parent(candidates) if level == 1 else self._chunk(candidates, self.K)
            
            # Move GT's group to front (enables early break, doesn't affect results)
            if gt_alive:
                for i, g in enumerate(groups):
                    if any(c['paper_id'] == gt_paper_id for c in g):
                        groups.insert(0, groups.pop(i))
                        break
            
            winners = []
            for group in groups:
                gt_in_group = gt_alive and any(c['paper_id'] == gt_paper_id for c in group)
                
                if len(group) == 1:
                    winners.append(group[0])
                else:
                    winner_idx = self._select_winner(group, research_question, background_survey, previous_hypothesis)
                    inference_calls_before_break += 1
                    winners.append(group[winner_idx])
                    
                    # Track GT elimination
                    if gt_in_group and group[winner_idx]['paper_id'] != gt_paper_id:
                        gt_alive = False
                        gt_eliminated_level = level
                        break  # GT is out, no need to continue
            
            if not gt_alive:
                break  # Exit while loop
            
            candidates = winners
        
        # Result
        found = gt_alive  # GT survived all rounds = GT won
        gt_survived = level if found else (gt_eliminated_level - 1 if gt_eliminated_level > 0 else 0)
        
        # propose_rank = winners_above + avg_rank_in_level
        # - winners_above: number of papers that advanced past GT's elimination level
        # - avg_rank_in_level: (1 + level_total) / 2, assuming uniform distribution
        if found:
            propose_rank = 1
        else:
            winners_above = self.level_sizes[gt_eliminated_level]
            level_total = self.level_sizes[gt_eliminated_level - 1]
            propose_rank = winners_above + (1 + level_total) / 2
        
        return TournamentSearchResult(
            found=found,
            depth=gt_survived,
            inference_calls_before_break=inference_calls_before_break,
            path=[],
            sample_id=sample_id,
            gt_paper_id=gt_paper_id,
            gt_title=gt_title,
            search_time=time.time() - start_time,
            propose_rank=propose_rank
        )
    
    def evaluate(
        self,
        eval_samples: List[Dict],
        incremental_output: str,
        verbose: bool = True,
        num_workers: int = 1
    ) -> Dict:
        """Evaluate on samples."""
        results = []
        total = len(eval_samples)
        
        os.makedirs(os.path.dirname(incremental_output) or '.', exist_ok=True)
        out_file = open(incremental_output, 'a')
        
        def process(sample, idx):
            r = self.search(
                sample['research_question'],
                sample['background_survey'],
                sample['sample_id'],
                sample['gt_paper_id'],
                sample['gt_title'],
                sample.get('prev_hyp')
            )
            if verbose:
                status = "✓" if r.found else "✗"
                print(f"[{idx+1}/{total}] {status} calls={r.inference_calls_before_break} rank={r.propose_rank} "
                      f"depth={r.depth} {r.search_time:.1f}s | {sample['gt_title'][:50]}...")
            return r
        
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = {ex.submit(process, s, i): i for i, s in enumerate(eval_samples)}
                for f in as_completed(futures):
                    r = f.result()
                    results.append(r)
                    out_file.write(json.dumps(asdict(r)) + '\n')
                    out_file.flush()
        finally:
            out_file.close()
        
        # Sort by original order
        sample_order = {s['sample_id']: i for i, s in enumerate(eval_samples)}
        results.sort(key=lambda r: sample_order[r.sample_id])
        
        # Stats
        found = sum(1 for r in results if r.found)
        return {
            'num_samples': len(results),
            'found': found,
            'accuracy': found / len(results) if results else 0,
            'avg_inference_calls_before_break': sum(r.inference_calls_before_break for r in results) / len(results) if results else 0,
            'avg_propose_rank': sum(r.propose_rank for r in results) / len(results) if results else 0,
            'avg_depth': sum(r.depth for r in results) / len(results) if results else 0,
            'total_time': sum(r.search_time for r in results),
            # Global tournament config
            'num_papers': self.num_papers,
            'expected_calls_full_tournament': self.expected_calls,
            'level_sizes': self.level_sizes,
            'branching_factor': self.K,
            'results': [asdict(r) for r in results]
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
        motivation_option: 0=no motivation, 
                          1=simple motivation (from inspiration['motivation']),
                          2=detailed motivation (parsed from hypothesis_components).
        truncate_survey: 0=no truncation, 1=truncate background_survey before problem/limitation
                        keywords (e.g., "however", "limitation", "gap").
    """
    title_to_pid = {p['title'].lower().strip(): pid for pid, p in tree.papers.items()}
    
    samples = []
    skipped = 0
    truncated_count = 0
    for ef in sorted(glob(os.path.join(eval_dir, "*.json"))):
        with open(ef) as f:
            data = json.load(f)
        
        hyp_comps = data['hypothesis_components']
        for idx, insp in enumerate(data['inspiration']):
            gt_title = insp['found_title'].lower().strip()
            if gt_title not in title_to_pid:
                # Paper not in corpus - skip (data mismatch, not a bug)
                skipped += 1
                continue
            
            prev_hyp = "\n\n".join(hyp_comps[str(j)] for j in range(idx)) if idx > 0 else None
            bg = data['background_survey']
            
            # Optionally truncate background before problem descriptions
            if truncate_survey:
                bg, was_truncated, _ = truncate_before_problem(bg)
                if was_truncated:
                    truncated_count += 1
            
            if motivation_option == 1:
                # Simple motivation from inspiration['motivation']
                # Append without strong directive label to avoid biasing model
                simple_mot = insp.get('motivation', '')
                if simple_mot:
                    bg = f"{bg}\n\n{simple_mot}"
            elif motivation_option == 2:
                # Detailed motivation parsed from hypothesis_components
                # Append without strong directive label
                hyp_comp = hyp_comps.get(str(idx), '')
                detailed_mot = parse_motivation_from_hypothesis_component(hyp_comp)
                if detailed_mot:
                    bg = f"{bg}\n\n{detailed_mot}"
            
            samples.append({
                'sample_id': f"{os.path.basename(ef).replace('.json', '')}_insp{idx}",
                'research_question': data['research_question'],
                'background_survey': bg,
                'prev_hyp': prev_hyp,
                'gt_title': insp['found_title'],
                'gt_paper_id': title_to_pid[gt_title]
            })
        
        if max_samples > 0 and len(samples) >= max_samples:
            break
    
    if skipped > 0:
        print(f"  Warning: skipped {skipped} samples (paper not in corpus)")
    if truncate_survey:
        print(f"  Survey truncation: {truncated_count}/{len(samples)} samples truncated")
    return samples[:max_samples] if max_samples > 0 else samples


def print_summary(results: List[Dict]) -> Dict:
    """Print and return summary stats."""
    if not results:
        return {}
    
    found = sum(1 for r in results if r['found'])
    summary = {
        'num_samples': len(results),
        'found': found,
        'accuracy': found / len(results),
        'avg_inference_calls_before_break': sum(r['inference_calls_before_break'] for r in results) / len(results),
        'avg_propose_rank': sum(r['propose_rank'] for r in results) / len(results),
        'avg_depth': sum(r['depth'] for r in results) / len(results),
    }
    
    print("\n" + "="*50)
    print("Tournament Search Summary")
    print("="*50)
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tournament Search Baseline")
    parser.add_argument("--tree-dir", default="${SEARCH_TREE_DIR}")
    parser.add_argument("--eval-dir", default="${TOMATO_STAR_EVAL_DIR}")
    parser.add_argument("--sglang-urls", nargs="+", default=["http://127.0.0.1:30000/v1"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=20480)
    parser.add_argument("--top-logprobs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--motivation-option", type=int, default=0, choices=[0, 1, 2],
                       help="0=no motivation, 1=simple (Problem/Gap), 2=detailed (full Motivation WHY)")
    parser.add_argument("--truncate-survey", type=int, default=0, choices=[0, 1],
                       help="0=no truncation, 1=truncate survey before problem keywords")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    print("="*50)
    print("Tournament Search (Baseline)")
    print("="*50)
    print(f"Visits ALL leaves: O(N/K) vs Hierarchical O(log N)")
    if args.truncate_survey:
        print("*** Survey: TRUNCATED (remove implicit motivation) ***")
    if args.motivation_option == 1:
        print("*** Motivation Option: SIMPLE (Problem/Gap + Solution Direction) ***")
    elif args.motivation_option == 2:
        print("*** Motivation Option: DETAILED (full Motivation WHY section) ***")
    
    evaluator = TournamentSearchEvaluator(
        args.tree_dir, args.sglang_urls,
        args.softmax_temperature, args.max_tokens, args.top_logprobs
    )
    
    samples = load_eval_samples(
        args.eval_dir, evaluator.tree, args.max_samples, 
        args.motivation_option, args.truncate_survey
    )
    print(f"Loaded {len(samples)} samples")
    
    # Resume handling
    os.makedirs(args.output_dir, exist_ok=True)
    inc_path = os.path.join(args.output_dir, "results_incremental.jsonl")
    prev_results = []
    
    if os.path.exists(inc_path):
        if not args.resume:
            raise RuntimeError(f"Results exist at {inc_path}. Use --resume or delete.")
        
        done_ids = set()
        with open(inc_path) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        done_ids.add(r['sample_id'])
                        prev_results.append(r)
                    except:
                        pass
        samples = [s for s in samples if s['sample_id'] not in done_ids]
        print(f"Resume: {len(prev_results)} done, {len(samples)} remaining")
        
        if not samples:
            summary = print_summary(prev_results)
            # Add global tournament config
            summary['num_papers'] = evaluator.num_papers
            summary['expected_calls_full_tournament'] = evaluator.expected_calls
            summary['level_sizes'] = evaluator.level_sizes
            summary['branching_factor'] = evaluator.K
            with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
            sys.exit(0)
    
    # Run
    result = evaluator.evaluate(samples, inc_path, verbose=True, num_workers=args.num_workers)
    
    # Merge & save
    all_results = prev_results + result.pop('results', [])
    summary = print_summary(all_results)
    # Add global tournament config
    summary['num_papers'] = evaluator.num_papers
    summary['expected_calls_full_tournament'] = evaluator.expected_calls
    summary['level_sizes'] = evaluator.level_sizes
    summary['branching_factor'] = evaluator.K
    
    with open(os.path.join(args.output_dir, "results.jsonl"), 'w') as f:
        for r in all_results:
            f.write(json.dumps(r) + '\n')
    with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output_dir}")
