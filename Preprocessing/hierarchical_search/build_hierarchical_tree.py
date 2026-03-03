#!/usr/bin/env python3
"""
Build Hierarchical Search Tree for Scientific Inspiration Retrieval

This script builds a multi-level tree structure from scientific papers using:
1. SPECTER2 embeddings for semantic representation
2. Bottom-up K-Means clustering with balanced assignment
3. Centroid paper selection (medoid) for node representation

The resulting tree enables efficient hierarchical search with O(log N) complexity.
Each leaf node contains exactly 1 paper, and each internal node has at most K children.

Reference: MOOSE-Star Paper, Section 3.4 "Hierarchical Search & Bounded Composition"
"""

import os
import json
import argparse
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Paper:
    """
    Represents a scientific paper.
    
    Attributes:
        paper_id: Content-based hash ID (NOT PMID). Generated from title+abstract
                  using MD5 hash. Example: "a3f2b1c9d0e4" (12 chars).
                  This ensures same paper always gets same ID regardless of source.
        title: Paper title.
        abstract: Paper abstract.
        year: Publication year (optional).
    """
    paper_id: str      # Content hash, NOT PMID. See generate_paper_id()
    title: str
    abstract: str
    year: Optional[int] = None
    
    def get_specter_input(self) -> str:
        """Format paper for SPECTER2 embedding: Title + [SEP] + Abstract"""
        return f"{self.title} [SEP] {self.abstract}"


@dataclass 
class TreeNode:
    """
    Represents a node in the hierarchical search tree.
    
    Tree Structure Example (bottom_up mode with 3035 papers):
    
        Level 0 (Root):    1 node    - represents ALL 3035 papers
                           │
        Level 1:          14 nodes   - each represents ~217 papers
                           │
        Level 2:         203 nodes   - each represents ~15 papers  
                           │
        Level 3 (Leaves): 3035 nodes - each represents 1 paper
    
    Attributes:
        node_id: Unique identifier for THIS NODE in the tree structure.
                 Format: "N{counter}". Example: "N145".
                 This is for tree navigation, NOT a paper ID.
        
        level: Depth in tree. Root = 0, increases toward leaves.
        
        is_leaf: True if this is a leaf node (contains actual papers).
        
        centroid_paper_id: The paper that best represents THIS NODE's cluster.
                          - For internal nodes: the most "central" paper among 
                            all papers in this subtree (selected by medoid algorithm)
                          - For leaf nodes: same as the single paper in paper_ids
                          This is what the IR model sees when choosing branches.
                          NOTE: This is a Paper ID (content hash), not a node_id.
        
        centroid_title: Title of the centroid paper (cached for quick display).
        
        children: Child TreeNodes (empty for leaf nodes).
                  For internal nodes: list of up to K=15 child nodes.
        
        paper_id: The Paper ID of the paper in this leaf node.
                  - For LEAF nodes: the single paper's ID (content hash)
                  - For INTERNAL nodes: None (papers are in descendants)
                  NOTE: This is a Paper ID (content hash), not a node_id.
        
        num_papers: Total count of papers in this subtree.
                    - For leaves: 1
                    - For internal: sum of all descendant papers
        
        parent_id: node_id of parent node (None for root).
    
    ID Types Summary:
        - node_id:           Tree structure ID    "N145"
        - paper_id:          Content hash         "a3f2b1c9d0e4" (leaf only)
        - centroid_paper_id: A paper_id           "a3f2b1c9d0e4"
        - parent_id:         A node_id            "N12"
    """
    node_id: str                         # Tree node ID, e.g., "L2_N145"
    level: int                           # Depth: root=0, leaves=max
    is_leaf: bool                        # True for leaf nodes
    centroid_paper_id: str               # Paper ID of representative paper for THIS node
    centroid_title: str                  # Title of centroid paper (for display)
    children: List['TreeNode'] = field(default_factory=list)    # Child nodes (internal only)
    paper_id: Optional[str] = None       # Paper ID (leaf only, None for internal nodes)
    num_papers: int = 0                  # Total papers in this subtree
    parent_id: Optional[str] = None      # Parent's node_id
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'node_id': self.node_id,
            'level': self.level,
            'is_leaf': self.is_leaf,
            'centroid_paper_id': self.centroid_paper_id,
            'centroid_title': self.centroid_title,
            'children': [child.to_dict() for child in self.children],
            'paper_id': self.paper_id,
            'num_papers': self.num_papers,
            'parent_id': self.parent_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TreeNode':
        """Reconstruct from dictionary."""
        children = [cls.from_dict(c) for c in data.get('children', [])]
        return cls(
            node_id=data['node_id'],
            level=data['level'],
            is_leaf=data['is_leaf'],
            centroid_paper_id=data['centroid_paper_id'],
            centroid_title=data['centroid_title'],
            children=children,
            paper_id=data.get('paper_id'),
            num_papers=data.get('num_papers', 0),
            parent_id=data.get('parent_id')
        )


# ============================================================================
# SPECTER2 Embedding Model
# ============================================================================

class SPECTER2Embedder:
    """SPECTER2 embedding model for scientific papers."""
    
    def __init__(self, device: str = None, batch_size: int = 32):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load SPECTER2 model and tokenizer."""
        from transformers import AutoTokenizer, AutoModel
        
        print(f"Loading SPECTER2 model on {self.device}...")
        model_name = "allenai/specter2_base"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("SPECTER2 model loaded successfully.")
        
    def embed_papers(self, papers: List[Paper], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of papers.
        
        Args:
            papers: List of Paper objects
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (num_papers, embedding_dim)
        """
        if self.model is None:
            self.load_model()
            
        texts = [p.get_specter_input() for p in papers]
        all_embeddings = []
        
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")
            
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings (CLS token)
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
                
        return np.vstack(all_embeddings)


# ============================================================================
# Hierarchical Tree Builder
# ============================================================================

class HierarchicalTreeBuilder:
    """
    Build hierarchical search tree using bottom-up K-Means clustering.
    
    Algorithm:
    1. Create N leaf nodes (1 paper per leaf)
    2. Use K-Means to group leaves into ~N/K parent nodes
    3. Apply balanced assignment to ensure each parent has ≤K children
    4. Recursively merge until reaching a single root node
    
    Properties:
    - Each leaf contains exactly 1 paper
    - Each internal node has at most K children (default K=15)
    - Tree depth = ceil(log_K(N)) ≈ 3 for N=3000, K=15
    - Optimal for 15-select-1 IR model
    """
    
    def __init__(
        self,
        branching_factor: int = 15,
        use_medoid: bool = True,
        random_state: int = 42
    ):
        """
        Args:
            branching_factor: Maximum children per node (K). Default 15 matches
                              the IR model's 15-select-1 capacity.
            use_medoid: If True, select centroid as the paper with highest avg
                        similarity to others (medoid). If False, use closest to mean.
            random_state: Random seed for reproducibility.
        """
        self.K = branching_factor
        self.use_medoid = use_medoid
        self.random_state = random_state
        self.node_counter = 0
        
    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        self.node_counter += 1
        return f"N{self.node_counter}"
    
    def _select_centroid(self, embeddings: np.ndarray) -> int:
        """
        Select the most representative paper (centroid) for a cluster.
        
        Uses medoid selection: finds the paper with highest average cosine
        similarity to all other papers in the cluster. This paper best
        represents the "center" of the cluster.
        
        Args:
            embeddings: Cluster embeddings (N x D)
            
        Returns:
            Index of the centroid paper
        """
        n = len(embeddings)
        if n == 1:
            return 0
            
        if self.use_medoid:
            # Medoid selection: paper with highest average similarity to others
            # For large clusters, use sampling to avoid OOM
            MAX_MEDOID_SIZE = 5000
            if n > MAX_MEDOID_SIZE:
                # Sample a subset and find medoid within it
                sample_idx = np.random.choice(n, MAX_MEDOID_SIZE, replace=False)
                sample_emb = embeddings[sample_idx]
                sim_matrix = cosine_similarity(sample_emb)
                avg_sim = sim_matrix.mean(axis=1)
                return int(sample_idx[np.argmax(avg_sim)])
            else:
                sim_matrix = cosine_similarity(embeddings)
                avg_sim = sim_matrix.mean(axis=1)
                return int(np.argmax(avg_sim))
        else:
            # Closest to mean embedding
            mean_emb = embeddings.mean(axis=0, keepdims=True)
            similarities = cosine_similarity(embeddings, mean_emb)
            return int(np.argmax(similarities.flatten()))
    
    def build_tree(
        self,
        papers: List[Paper],
        embeddings: np.ndarray
    ) -> TreeNode:
        """
        Build tree bottom-up: each paper is a leaf, then merge upward.
        
        Structure:
        - Level N (leaves): Each leaf = 1 paper (total = N papers)
        - Level N-1: ~N/K nodes, each with ~K children
        - ...
        - Level 0 (root): 1 node with ~K children
        
        Key constraint: Each node has AT MOST K children.
        Uses balanced assignment to ensure this without increasing tree depth.
        """
        from sklearn.cluster import KMeans
        
        n_papers = len(papers)
        print(f"  Creating {n_papers} leaf nodes (1 paper per leaf)...")
        
        # Step 1: Create leaf nodes - each paper is a leaf
        leaves = []
        for i, paper in enumerate(papers):
            leaf_node = TreeNode(
                node_id=self._generate_node_id(),
                level=0,  # Will be updated later
                is_leaf=True,
                centroid_paper_id=paper.paper_id,
                centroid_title=paper.title,
                children=[],
                paper_id=paper.paper_id,  # Single paper
                num_papers=1,
                parent_id=None
            )
            leaves.append(leaf_node)
        
        leaf_embeddings = embeddings.copy()
        print(f"  Created {len(leaves)} leaves (1 paper each)")
        
        # Step 2: Recursively merge leaves into parent nodes
        current_level_nodes = leaves
        current_level_embeddings = leaf_embeddings
        level = 0
        
        while len(current_level_nodes) > 1:
            level += 1
            n_nodes = len(current_level_nodes)
            
            # Use ceiling division to get enough clusters
            n_parents = max(1, (n_nodes + self.K - 1) // self.K)
            
            print(f"  Level {level}: merging {n_nodes} nodes into {n_parents} parents...")
            
            # K-Means to group nodes
            if n_parents == 1:
                parent_assignments = np.zeros(n_nodes, dtype=int)
            else:
                kmeans = KMeans(
                    n_clusters=n_parents,
                    random_state=self.random_state,
                    n_init=10
                )
                parent_assignments = kmeans.fit_predict(current_level_embeddings)
                
                # Balance assignments: ensure no cluster > K
                # parent_assignments: [n_nodes]; each element is the index of the cluster
                parent_assignments = self._balance_assignments(
                    parent_assignments, 
                    current_level_embeddings, # [n_nodes, embedding_dim]
                    kmeans.cluster_centers_ # [n_clusters, embedding_dim]
                )
            
            # Create parent nodes
            next_level_nodes = []
            next_level_embeddings = []
            
            for parent_id in range(n_parents):
                mask = parent_assignments == parent_id
                children = [n for n, m in zip(current_level_nodes, mask) if m] 
                child_embs = current_level_embeddings[mask]
                
                if len(children) == 0:
                    continue
                
                parent_node, parent_emb = self._create_parent_node(
                    children, child_embs, level
                )
                next_level_nodes.append(parent_node)
                next_level_embeddings.append(parent_emb)
            
            # Report cluster size distribution
            sizes = [len(n.children) for n in next_level_nodes if not n.is_leaf]
            if sizes:
                print(f"    -> {len(next_level_nodes)} parents, children per parent: {min(sizes)}-{max(sizes)} (avg {np.mean(sizes):.1f})")
            
            current_level_nodes = next_level_nodes
            current_level_embeddings = np.array(next_level_embeddings) if next_level_embeddings else np.array([])
        
        # Return root and fix levels (root=0, leaves=max_depth)
        root = current_level_nodes[0]
        self._fix_levels(root, level=0)
        
        return root
    
    def _balance_assignments(
        self,
        assignments: np.ndarray,
        embeddings: np.ndarray,
        centers: np.ndarray
    ) -> np.ndarray:
        """
        Rebalance cluster assignments to ensure no cluster has more than K members.
        
        Strategy: Move nodes from oversized clusters to their next-best cluster
        that still has room.
        """
        assignments = assignments.copy()
        n_clusters = len(centers)
        
        # Compute similarities from each point to all centers
        similarities = cosine_similarity(embeddings, centers)  # (n_nodes, n_clusters)
        
        # Sort each node's cluster preferences (highest similarity first)
        preferences = np.argsort(-similarities, axis=1)  # (n_nodes, n_clusters)
        
        # Iteratively fix oversized clusters
        max_iterations = 100
        for iteration in range(max_iterations):
            # Count cluster sizes
            cluster_sizes = np.bincount(assignments, minlength=n_clusters)
            
            # Find oversized clusters
            oversized = np.where(cluster_sizes > self.K)[0]
            if len(oversized) == 0:
                break  # All clusters are <= K
            
            # For each oversized cluster, move excess nodes
            for cluster_id in oversized:
                excess = cluster_sizes[cluster_id] - self.K
                if excess <= 0:
                    continue
                
                # Find nodes in this cluster, sorted by distance to center (furthest first)
                in_cluster = np.where(assignments == cluster_id)[0]
                cluster_sims = similarities[in_cluster, cluster_id]
                # Move the nodes that are furthest from center
                move_order = in_cluster[np.argsort(cluster_sims)]  # Ascending = furthest first
                
                moved = 0
                for node_idx in move_order:
                    if moved >= excess:
                        break
                    
                    # Find best alternative cluster with room
                    for alt_cluster in preferences[node_idx]:
                        if alt_cluster == cluster_id:
                            continue
                        if cluster_sizes[alt_cluster] < self.K:
                            # Move to this cluster
                            assignments[node_idx] = alt_cluster
                            cluster_sizes[cluster_id] -= 1
                            cluster_sizes[alt_cluster] += 1
                            moved += 1
                            break
        
        return assignments
    
    def _create_parent_node(
        self,
        children: List[TreeNode],
        child_embs: np.ndarray,
        level: int
    ) -> Tuple[TreeNode, np.ndarray]:
        """Create a parent node from a list of children (must be <= K children)."""
        parent_node_id = self._generate_node_id()
        total_papers = sum(c.num_papers for c in children)
        
        for child in children:
            child.parent_id = parent_node_id
        
        # Select centroid from children's centroids (medoid selection)
        centroid_idx = self._select_centroid(child_embs)
        centroid_child = children[centroid_idx]
        
        parent_node = TreeNode(
            node_id=parent_node_id,
            level=level,
            is_leaf=False,
            centroid_paper_id=centroid_child.centroid_paper_id,
            centroid_title=centroid_child.centroid_title,
            children=children,
            paper_id=None,  # Internal nodes don't have a paper_id
            num_papers=total_papers,
            parent_id=None
        )
        
        return parent_node, child_embs[centroid_idx]
    
    def _fix_levels(self, node: TreeNode, level: int):
        """Set correct levels: root=0, increasing toward leaves."""
        node.level = level
        for child in node.children:
            self._fix_levels(child, level + 1)


# ============================================================================
# Data Loading
# ============================================================================

def generate_paper_id(title: str, abstract: str) -> str:
    """
    Generate a unique paper ID from title and abstract.
    
    NOTE: This is a content-based hash, NOT a PMID or DOI.
    We use content hashing because:
    1. Input data may lack consistent PMIDs
    2. Same paper always gets same ID regardless of source
    3. Automatically deduplicates identical papers
    
    Args:
        title: Paper title
        abstract: Paper abstract
        
    Returns:
        12-character hex string, e.g., "a3f2b1c9d0e4"
    """
    content = f"{title}|||{abstract}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def compute_papers_hash(papers: List[Paper]) -> str:
    """Compute a hash of all papers to detect changes."""
    # Sort by paper_id for deterministic hash
    sorted_ids = sorted([p.paper_id for p in papers])
    content = "|".join(sorted_ids)
    return hashlib.md5(content.encode()).hexdigest()[:16]


def load_inspirations_from_sft_qa_dir(sft_qa_data_dir: str) -> List[Paper]:
    """
    Load all unique inspirations from SFT QA data directory.
    
    Args:
        sft_qa_data_dir: Path to directory containing SFT QA JSON files
        
    Returns:
        List of Paper objects (deduplicated by title)
    """
    print(f"Loading inspirations from: {sft_qa_data_dir}")
    
    papers_dict = {}  # title -> Paper (for deduplication)
    
    # Sort for deterministic order (same title may appear in multiple files)
    files = sorted([f for f in os.listdir(sft_qa_data_dir) if f.endswith('.json')])
    
    for filename in tqdm(files, desc="Loading papers"):
        filepath = os.path.join(sft_qa_data_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {filename}: {e}")
            continue
            
        # Extract year from filename (format: YYYY_PMID.json)
        try:
            year_str = filename.split('_')[0]
            year = int(year_str) if year_str != "0000" else 2020
        except (ValueError, IndexError):
            year = None
            
        # Add inspirations
        if 'inspiration' in data:
            for insp in data['inspiration']:
                title = insp.get('found_title', '').strip()
                abstract = insp.get('found_abstract', '').strip()
                
                if not title or not abstract:
                    continue
                    
                # Use title as dedup key (lowercased)
                title_key = title.lower()
                
                if title_key not in papers_dict:
                    paper_id = generate_paper_id(title, abstract)
                    papers_dict[title_key] = Paper(
                        paper_id=paper_id,
                        title=title,
                        abstract=abstract,
                        year=year  # From filename (YYYY_PMID.json)
                    )
    
    papers = list(papers_dict.values())
    print(f"Loaded {len(papers)} unique inspirations")
    return papers


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build Hierarchical Search Tree for Scientific Papers"
    )
    
    # Input
    parser.add_argument(
        '--sft_qa_dir',
        type=str,
        required=True,
        help='Path to SFT QA data directory (extracts inspirations from JSON files)'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for tree and embeddings'
    )
    
    # Tree building options
    parser.add_argument(
        '--branching_factor',
        type=int,
        default=15,
        help='Max children per node, should match IR model capacity (default: 15)'
    )
    parser.add_argument(
        '--use_medoid',
        type=int,
        default=1,
        help='Use medoid selection (1) or closest to mean (0)'
    )
    
    # Embedding options
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for embedding generation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device for embedding model (default: auto)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load papers
    papers = load_inspirations_from_sft_qa_dir(args.sft_qa_dir)
    
    if len(papers) == 0:
        print("Error: No papers loaded!")
        return
    
    print(f"\nTotal papers: {len(papers)}")
    
    # Compute hash for cache validation
    papers_hash = compute_papers_hash(papers)
    print(f"Papers hash: {papers_hash}")
    
    # File paths
    embeddings_file = os.path.join(args.output_dir, 'embeddings.npy')
    papers_file = os.path.join(args.output_dir, 'papers.json')
    hash_file = os.path.join(args.output_dir, 'papers_hash.txt')
    
    # Check if cached embeddings are valid
    cached_hash = None
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            cached_hash = f.read().strip()
    
    if os.path.exists(embeddings_file) and cached_hash == papers_hash:
        # Valid cache - hash matches
        print(f"Loading cached embeddings (hash verified): {embeddings_file}")
        embeddings = np.load(embeddings_file)
    else:
        if os.path.exists(embeddings_file) and cached_hash != papers_hash:
            print(f"⚠️ Papers changed (hash mismatch), regenerating embeddings...")
        # Generate new embeddings
        embedder = SPECTER2Embedder(
            device=args.device,
            batch_size=args.batch_size
        )
        embeddings = embedder.embed_papers(papers)
        
        # Save embeddings and hash
        np.save(embeddings_file, embeddings)
        with open(hash_file, 'w') as f:
            f.write(papers_hash)
        print(f"Saved embeddings to: {embeddings_file}")
    
    # Final validation
    if embeddings.shape[0] != len(papers):
        raise ValueError(f"CRITICAL: Embedding count ({embeddings.shape[0]}) != paper count ({len(papers)})")
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save papers metadata
    papers_data = [
        {
            'paper_id': p.paper_id,
            'title': p.title,
            'abstract': p.abstract,
            'year': p.year
        }
        for p in papers
    ]
    with open(papers_file, 'w') as f:
        json.dump(papers_data, f, indent=2)
    print(f"Saved papers metadata to: {papers_file}")
    
    # Build tree
    print(f"\nBuilding hierarchical tree...")
    builder = HierarchicalTreeBuilder(
        branching_factor=args.branching_factor,
        use_medoid=bool(args.use_medoid),
        random_state=args.seed
    )
    tree = builder.build_tree(papers, embeddings)
    
    # Save tree
    tree_file = os.path.join(args.output_dir, 'hierarchical_tree.json')
    with open(tree_file, 'w') as f:
        json.dump(tree.to_dict(), f, indent=2)
    print(f"Saved tree to: {tree_file}")
    
    # Print tree statistics
    def count_nodes(node: TreeNode) -> Tuple[int, int, int]:
        """Count total nodes, leaf nodes, and max depth."""
        if node.is_leaf:
            return 1, 1, node.level
        
        total, leaves, max_depth = 1, 0, node.level
        for child in node.children:
            t, l, d = count_nodes(child)
            total += t
            leaves += l
            max_depth = max(max_depth, d)
        return total, leaves, max_depth
    
    total_nodes, leaf_nodes, max_depth = count_nodes(tree)
    
    print("\n" + "="*50)
    print("Tree Statistics:")
    print("="*50)
    print(f"Total papers: {len(papers)}")
    print(f"Total nodes: {total_nodes}")
    print(f"Leaf nodes: {leaf_nodes}")
    print(f"Internal nodes: {total_nodes - leaf_nodes}")
    print(f"Max depth: {max_depth}")
    print(f"Branching factor: {args.branching_factor}")
    print(f"Avg papers per leaf: {len(papers) / leaf_nodes:.1f}")
    print("="*50)
    
    # Save config
    config = {
        'branching_factor': args.branching_factor,
        'use_medoid': args.use_medoid,
        'seed': args.seed,
        'num_papers': len(papers),
        'embedding_dim': embeddings.shape[1],
        'total_nodes': total_nodes,
        'leaf_nodes': leaf_nodes,
        'max_depth': max_depth
    }
    config_file = os.path.join(args.output_dir, 'tree_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_file}")


if __name__ == '__main__':
    main()

