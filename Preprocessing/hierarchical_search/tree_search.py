#!/usr/bin/env python3
"""
Hierarchical Tree Search for Scientific Inspiration Retrieval

Usage:
    from tree_search import HierarchicalSearchTree
    
    tree = HierarchicalSearchTree.load('/path/to/tree_dir')
    candidates = tree.get_candidates_at_node(tree.root)
    
    # After IR model selects index i
    next_node = tree.get_child_by_index(tree.root, i)
"""

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Candidate:
    """A candidate for IR model selection."""
    index: int          # 0 to K-1
    node_id: str        # Tree node ID
    paper_id: str       # Centroid paper ID  
    title: str
    abstract: str
    num_papers: int     # Papers in this subtree
    is_leaf: bool


class HierarchicalSearchTree:
    """Hierarchical tree for O(log N) inspiration retrieval."""
    
    def __init__(self, tree_dict: Dict, papers_dict: Dict[str, Dict], config: Dict):
        self.root = tree_dict
        self.papers = papers_dict
        self.config = config
        self._node_index = {}
        self._build_index(self.root)
        
    def _build_index(self, node: Dict):
        """Build node lookup index."""
        self._node_index[node['node_id']] = node
        for child in node.get('children', []):
            self._build_index(child)
    
    @classmethod
    def load(cls, tree_dir: str, validate: bool = True) -> 'HierarchicalSearchTree':
        """Load tree from directory.
        
        Args:
            tree_dir: Directory containing tree files
            validate: If True, verify data consistency
        """
        with open(os.path.join(tree_dir, 'hierarchical_tree.json')) as f:
            tree_dict = json.load(f)
        with open(os.path.join(tree_dir, 'papers.json')) as f:
            papers_dict = {p['paper_id']: p for p in json.load(f)}
        with open(os.path.join(tree_dir, 'tree_config.json')) as f:
            config = json.load(f)
        
        instance = cls(tree_dict, papers_dict, config)
        
        if validate:
            instance._validate_consistency()
        
        return instance
    
    def _validate_consistency(self):
        """Verify tree and papers data are consistent."""
        missing_papers = []
        
        def check_node(node):
            # Check centroid
            if node.get('centroid_paper_id') and node['centroid_paper_id'] not in self.papers:
                missing_papers.append(node['centroid_paper_id'])
            # Check leaf paper
            if node.get('is_leaf') and node.get('paper_id'):
                if node['paper_id'] not in self.papers:
                    missing_papers.append(node['paper_id'])
            # Check children
            for child in node.get('children', []):
                check_node(child)
        
        check_node(self.root)
        
        if missing_papers:
            raise ValueError(
                f"Data inconsistency: {len(missing_papers)} paper IDs in tree not found in papers.json. "
                f"First few: {missing_papers[:5]}"
            )
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node by ID."""
        return self._node_index.get(node_id)
    
    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """Get paper metadata."""
        return self.papers.get(paper_id)
    
    def get_candidates_at_node(self, node: Dict) -> List[Candidate]:
        """
        Get candidates at a node for IR model.
        
        For internal nodes: returns children (with centroid papers)
        For leaf nodes: returns the single paper in this leaf
        """
        candidates = []
        
        if node.get('is_leaf'):
            # Leaf: return the single paper
            paper_id = node.get('paper_id')
            if not paper_id:
                raise ValueError(f"Leaf node {node.get('node_id')} has no paper_id")
            if paper_id not in self.papers:
                raise KeyError(f"Leaf paper {paper_id} not found in papers dict")
            paper = self.papers[paper_id]
            candidates.append(Candidate(
                index=0,
                node_id=node['node_id'],
                paper_id=paper_id,
                title=paper['title'],
                abstract=paper['abstract'],
                num_papers=1,
                is_leaf=True
            ))
        else:
            # Internal: return children
            for i, child in enumerate(node.get('children', [])):
                centroid_id = child['centroid_paper_id']
                if centroid_id not in self.papers:
                    raise KeyError(f"Centroid paper {centroid_id} not found in papers dict")
                paper = self.papers[centroid_id]
                candidates.append(Candidate(
                    index=i,
                    node_id=child['node_id'],
                    paper_id=centroid_id,
                    title=paper['title'],  # Required field
                    abstract=paper['abstract'],  # Required field
                    num_papers=child.get('num_papers', 0),
                    is_leaf=child.get('is_leaf', False)
                ))
        
        return candidates
    
    def get_child_by_index(self, node: Dict, index: int) -> Optional[Dict]:
        """Get child node by index (0 to K-1)."""
        children = node.get('children', [])
        return children[index] if 0 <= index < len(children) else None
    
    def navigate_with_auto_skip(self, node: Dict, index: int) -> tuple[Dict, List[Dict]]:
        """
        Navigate to child, automatically skipping nodes with only 1 child.
        
        This saves inference steps - if a node has only 1 child, there's no 
        choice to make, so we skip directly to that child.
        
        Returns:
            (final_node, path): The final node and the path taken (including skipped nodes)
        """
        child = self.get_child_by_index(node, index)
        if child is None:
            return node, [node]
        
        path = [child]
        
        # Auto-skip single-child nodes
        while not child.get('is_leaf') and len(child.get('children', [])) == 1:
            child = child['children'][0]
            path.append(child)
        
        return child, path
    
    def needs_inference(self, node: Dict) -> bool:
        """
        Check if inference is needed at this node.
        
        Returns False if:
        - Node is a leaf (search complete)
        - Node has only 1 child (no choice to make)
        """
        if node.get('is_leaf'):
            return False
        children = node.get('children', [])
        return len(children) > 1
    
    def format_candidates(
        self,
        candidates: List[Candidate],
        include_abstract: bool = True,
        max_abstract_len: int = 500
    ) -> str:
        """Format candidates as numbered list for IR model."""
        lines = []
        for c in candidates:
            if include_abstract:
                abstract = c.abstract[:max_abstract_len] + '...' if len(c.abstract) > max_abstract_len else c.abstract
                lines.append(f"{c.index + 1}. [{c.title}] {abstract}")
            else:
                lines.append(f"{c.index + 1}. {c.title}")
        return "\n".join(lines)
    
    def get_all_leaf_papers(self) -> List[str]:
        """Get all paper IDs from leaf nodes."""
        papers = []
        def collect(node):
            if node.get('is_leaf'):
                paper_id = node.get('paper_id')
                if paper_id:
                    papers.append(paper_id)
            else:
                for child in node.get('children', []):
                    collect(child)
        collect(self.root)
        return papers
    
    def get_stats(self) -> Dict:
        """Get tree statistics."""
        def count(node):
            if node.get('is_leaf'):
                return 1, 1, node['level']
            total, leaves, depth = 1, 0, node['level']
            for child in node.get('children', []):
                t, l, d = count(child)
                total += t
                leaves += l
                depth = max(depth, d)
            return total, leaves, depth
        
        total, leaves, depth = count(self.root)
        return {
            'total_nodes': total,
            'leaf_nodes': leaves,
            'max_depth': depth,
            'num_papers': self.config.get('num_papers', 0),
            'branching_factor': self.config.get('branching_factor', 0)
        }
