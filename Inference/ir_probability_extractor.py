#!/usr/bin/env python3
"""
Inspiration Retrieval Probability Extractor

Extracts probability distribution over candidate selections from the trained IR model.
Designed for hierarchical tree search where we need selection probabilities.

Usage:
    extractor = IRProbabilityExtractor(base_urls=["http://localhost:30000/v1"])
    
    # Multiple endpoints for load balancing (round-robin)
    extractor = IRProbabilityExtractor(base_urls=[
        "http://node1:30000/v1",
        "http://node1:30001/v1",
        "http://node2:30000/v1",
    ])
    
    result = extractor.get_selection_probabilities(
        research_question="...",
        background_survey="...",
        candidates=[{"title": "...", "abstract": "..."}, ...],
    )
    # result.probabilities = {"A": 0.45, "B": 0.30, ...}

SGLang Deployment:
    # Single instance (4 GPUs)
    ./start_sglang.sh
    
    # Dual instances (8 GPUs, 2x throughput)
    ./start_sglang_dual.sh
"""

import os, time, httpx
import sys
import math
import threading
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Add parent paths for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

sys.path.insert(0, os.path.join(parent_dir, 'utils'))
from prompt_store import instruction_prompts

# Alphabetical labels (A-Z, typically A-O for 15 candidates)
LABELS = [chr(ord('A') + i) for i in range(26)]


@dataclass
class SelectionResult:
    """Result of IR model selection with probabilities."""
    probabilities: Dict[str, float]  # Label -> probability
    selected_label: str              # Argmax selection
    selected_index: int              # 0-indexed position
    num_candidates: int
    raw_logprobs: Optional[Dict[str, float]] = None
    response_text: Optional[str] = None  # Full response if requested


def build_ir_prompt(
    research_question: str,
    background_survey: str,
    candidates: List[Dict[str, str]],
    previous_hypothesis: Optional[str] = None
) -> Tuple[str, Dict[str, str]]:
    """
    Build IR prompt in the exact format used during training.
    
    Returns:
        (prompt_string, label_to_title_mapping)
    """
    prompts = instruction_prompts("inspiration_retrieval_with_reasoning_with_alphabetical_candidates")
    
    pre_hyp_text = previous_hypothesis if previous_hypothesis else "No previous hypothesis."
    
    if not candidates or len(candidates) < 2:
        raise ValueError(f"Need 2-26 candidates, got {len(candidates) if candidates else 0}")
    if len(candidates) > 26:
        raise ValueError(f"Too many candidates: {len(candidates)} (max 26)")
    
    # Format candidates with labels
    candidates_list = []
    label_to_title = {}
    for i, c in enumerate(candidates):
        label = LABELS[i]
        candidates_list.append(f"### Candidate [{label}]\n**Title:** {c['title']}\n**Abstract:** {c['abstract']}")
        label_to_title[label] = c['title']
    
    full_prompt = (
        prompts[0] + research_question + 
        prompts[1] + background_survey + 
        prompts[2] + pre_hyp_text + 
        prompts[3] + "\n\n".join(candidates_list) + 
        prompts[4]
    )
    
    return full_prompt, label_to_title


class IRProbabilityExtractor:
    """
    Extract selection probabilities from IR model via SGLang.
    
    The model outputs: <think>...</think>**Selected ID starts:** [X] **Selected ID ends**
    We extract logprobs at position X to get probability distribution over candidates.
    
    Example:
        # Single or multiple endpoints (round-robin load balancing)
        extractor = IRProbabilityExtractor(base_urls=[
            "http://node1:30000/v1",
            "http://node1:30001/v1",
        ])
    """
    
    def __init__(
        self, 
        base_urls: List[str],
        model_name: str = "default",
        api_key: str = "EMPTY"
    ):
        """
        Args:
            base_urls: SGLang endpoint URLs (list, supports load balancing if multiple)
            model_name: Model name for API
            api_key: API key (usually "EMPTY" for local SGLang)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        
        if not base_urls:
            raise ValueError("base_urls is required")
        
        self.base_urls = base_urls
        
        # Configure HTTP clients with larger connection pool for high concurrency
        # Default pool is too small (10-20), causing bottleneck with many workers

        def make_http_client():
            return httpx.Client(
                limits=httpx.Limits(
                    max_connections=1024,           # Max concurrent connections per endpoint
                    max_keepalive_connections=1024  # Keep all connections alive for reuse
                ),
                timeout=httpx.Timeout(3600.0, connect=30.0)  # 60 min read, 30s connect
            )
        
        # Each endpoint gets its own connection pool
        self.clients = [
            OpenAI(api_key=api_key, base_url=url, http_client=make_http_client()) 
            for url in self.base_urls
        ]
        self.model_name = model_name
        self._call_count = 0  # For round-robin
        self._lock = threading.Lock()  # Thread-safe counter
    
    def get_selection_probabilities(
        self,
        research_question: str,
        background_survey: str,
        candidates: List[Dict[str, str]],
        previous_hypothesis: Optional[str] = None,
        softmax_temperature: float = 1.0,
        generation_temperature: float = 0.0,
        max_tokens: int = 20480,  # 20k tokens, fit within 32k context with ~10k input
        top_logprobs: int = 30,  # Ensure all 15 candidate labels (A-O) are covered
        return_response: bool = False
    ) -> SelectionResult:
        """
        Get probability distribution over candidate selections.
        
        Args:
            research_question: The research question
            background_survey: Background survey/existing methods
            candidates: List of {"title": str, "abstract": str}
            previous_hypothesis: Previous hypothesis if exists
            softmax_temperature: Temperature for softmax scaling of output probabilities.
                                 Higher values (e.g., 10.0) produce flatter distributions.
                                 Default 1.0 uses raw logprobs without scaling.
            generation_temperature: Temperature for LLM text generation.
                                    0.0 = greedy/deterministic, >0 = more random.
            max_tokens: Maximum tokens for LLM output (24k for R1-style reasoning)
            top_logprobs: Number of top logprobs to return per token (should cover all candidates)
            return_response: If True, include full response text in result
            
        Returns:
            SelectionResult with probabilities and selection
        """
        num_candidates = len(candidates)
        valid_labels = LABELS[:num_candidates]
        
        # Build prompt
        prompt, _ = build_ir_prompt(
            research_question=research_question,
            background_survey=background_survey,
            candidates=candidates,
            previous_hypothesis=previous_hypothesis
        )
        
        # Call API with logprobs (with unified retry for API errors and format errors)
        max_retries = 10  # Total retries for any error (API or format)
        last_error = None
        response = None
        raw_logprobs = None
        
        for attempt in range(max_retries):
            # Thread-safe round-robin client selection for load balancing
            with self._lock:
                client = self.clients[self._call_count % len(self.clients)]
                self._call_count += 1
            
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=generation_temperature,
                    logprobs=True,
                    top_logprobs=top_logprobs,
                )
            except Exception as e:
                last_error = e
                error_str = str(e)
                # Simplify common errors for cleaner logs
                if "504" in error_str:
                    error_msg = "504 Gateway Timeout"
                elif "502" in error_str:
                    error_msg = "502 Bad Gateway"
                elif "503" in error_str:
                    error_msg = "503 Service Unavailable"
                elif "Connection" in error_str or "timeout" in error_str.lower():
                    error_msg = f"Connection error: {error_str[:100]}"
                else:
                    error_msg = f"{error_str[:150]}"
                print(f"\t[Retry {attempt+1}/{max_retries}] {error_msg}", file=sys.stderr)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    continue
                else:
                    raise RuntimeError(f"API failed after {max_retries} retries: {error_msg}") from last_error
            
            content = response.choices[0].message.content
            logprobs_data = response.choices[0].logprobs
            
            # Log token usage for performance monitoring
            if response.usage:
                output_tokens = response.usage.completion_tokens
                if output_tokens > 5000:  # Only warn for long outputs
                    print(f"\t[Perf] Long output: {output_tokens} tokens", file=sys.stderr)
            
            # Extract logprobs at label position
            raw_logprobs = self._extract_label_logprobs(content, logprobs_data, valid_labels)
            
            # Check if format error (all values are -1.0, indicating marker not found)
            is_format_error = all(v == -1.0 for v in raw_logprobs.values())
            if is_format_error and attempt < max_retries - 1:
                print(f"\t[Retry {attempt+1}/{max_retries}] Format error: selection marker not found, retrying...", file=sys.stderr)
                # print(f"\n\n\tContent: {content}\n\n", file=sys.stderr)
                time.sleep(1)  # Brief pause before retry
                continue
            break  # Success or max retries reached
        
        # Apply temperature scaling and softmax
        probabilities = self._logprobs_to_probs(raw_logprobs, softmax_temperature)
        
        # Get argmax
        selected_label = max(probabilities, key=probabilities.get)
        
        return SelectionResult(
            probabilities=probabilities,
            selected_label=selected_label,
            selected_index=LABELS.index(selected_label),
            num_candidates=num_candidates,
            raw_logprobs=raw_logprobs,
            response_text=content if return_response else None
        )
    
    def _extract_label_logprobs(
        self, 
        content: str, 
        logprobs_data, 
        valid_labels: List[str]
    ) -> Dict[str, float]:
        """
        Extract logprobs for candidate labels from the model response.
        
        Expected model output format (trained format):
            <think>...reasoning...</think>
            
            **Selected ID starts:** [X] **Selected ID ends**
            
            **Selection Reason starts:** ...reason... **Selection Reason ends**
        
        We locate the label token right after "**Selected ID starts:** [" and extract
        the logprobs for all valid candidate labels (A-O) at that position.
        
        Args:
            content: Generated text from model
            logprobs_data: OpenAI-style logprobs object with .content list
            valid_labels: Valid candidate labels, e.g. ['A','B',...,'O'] for 15 candidates
        
        Returns:
            Dict mapping each valid label to its logprob at the selection position.
            Labels not in top_logprobs get -100.0 (essentially zero probability).
            Example: {'A': -0.5, 'B': -2.3, 'C': -100.0, ...}
        
        Note:
            If the selection marker is not found (e.g., output truncated), returns
            uniform low logprobs (-1.0 for all) as a fallback.
        """
        if logprobs_data is None or logprobs_data.content is None:
            raise RuntimeError("No logprobs returned. Check SGLang server configuration.")
        
        # Find "**Selected ID starts:** [" in content
        marker = "**Selected ID starts:** ["
        marker_pos = content.find(marker)
        if marker_pos == -1:
            # Try fallback patterns (model may have slight variations)
            for alt in ["Selected ID starts:** [", "starts:** ["]:
                pos = content.find(alt)
                if pos != -1:
                    marker_pos = pos + len(alt) - 1  # Position at "["
                    break
        
        if marker_pos == -1:
            # Output was likely truncated before reaching the selection marker.
            # Return uniform low logprobs so softmax gives equal probabilities.
            # This is a graceful degradation - search can continue with random selection.
            # print(f"Warning: Selection marker not found (output truncated or format error)", file=sys.stderr)
            return {label: -1.0 for label in valid_labels}
        
        marker_end = marker_pos + len(marker)
        
        # Find the token index at marker_end position (the label token after "[")
        # logprobs_data.content is a list of token_info objects with .token and .logprob
        char_count = 0
        label_token_idx = None
        
        for i, token_info in enumerate(logprobs_data.content):
            next_count = char_count + len(token_info.token)
            if next_count > marker_end:
                # Found the position - determine which token is the label
                if token_info.token.strip() == "[":
                    label_token_idx = i + 1  # Label is next token
                elif any(label in token_info.token for label in valid_labels):
                    label_token_idx = i  # This token contains the label
                else:
                    label_token_idx = i + 1  # Assume label is next
                break
            char_count = next_count
        
        if label_token_idx is None or label_token_idx >= len(logprobs_data.content):
            raise RuntimeError(f"Could not locate label token position")
        
        # Extract logprobs from the label token position
        token_info = logprobs_data.content[label_token_idx]
        result = {label: -100.0 for label in valid_labels}  # Default: very low prob
        
        # Get logprobs from top_logprobs list
        if token_info.top_logprobs:
            for lp in token_info.top_logprobs:
                token = lp.token.strip()
                if token in valid_labels:
                    result[token] = lp.logprob
        
        # Also include the actual generated token's logprob
        actual = token_info.token.strip()
        if actual in valid_labels:
            result[actual] = token_info.logprob
        
        return result
    
    def _logprobs_to_probs(
        self, 
        logprobs: Dict[str, float], 
        softmax_temperature: float
    ) -> Dict[str, float]:
        """
        Convert logprobs to normalized probability distribution with temperature scaling.
        
        Formula: P(label) = softmax(logprob / softmax_temperature)
                          = exp(logprob / T) / sum(exp(logprobs / T))
        
        Args:
            logprobs: Dict mapping labels to log-probabilities.
                      Example: {'A': -0.5, 'B': -2.3, 'C': -100.0, ...}
            softmax_temperature: Temperature for softmax scaling.
                         - T=1.0: Use raw logprobs (model's actual distribution)
                         - T>1.0: Flatter distribution (more uniform, less confident)
                         - T<1.0: Sharper distribution (more peaked, more confident)
        
        Returns:
            Dict mapping labels to probabilities that sum to 1.0.
            Example: {'A': 0.65, 'B': 0.30, 'C': 0.05, ...}
        """
        if softmax_temperature != 1.0:
            logprobs = {k: v / softmax_temperature for k, v in logprobs.items()}
        
        # Numerically stable softmax: subtract max before exp
        max_lp = max(logprobs.values())
        exp_lp = {k: math.exp(v - max_lp) for k, v in logprobs.items()}
        total = sum(exp_lp.values())
        return {k: v / total for k, v in exp_lp.items()}


# =============================================================================
# Utility Functions
# =============================================================================

def top_k_labels(probs: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
    """Get top-k labels by probability."""
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:k]


def sample_from_probabilities(probs: Dict[str, float]) -> str:
    """Sample a label from probability distribution."""
    import numpy as np
    labels = list(probs.keys())
    p = [probs[l] for l in labels]
    return np.random.choice(labels, p=p)


# =============================================================================
# Example / Test
# =============================================================================

if __name__ == "__main__":
    # Test prompt building only (no model needed)
    test_candidates = [
        {"title": "Paper A", "abstract": "About neural networks..."},
        {"title": "Paper B", "abstract": "About graphs..."},
        {"title": "Paper C", "abstract": "About optimization..."},
    ]
    
    prompt, label_map = build_ir_prompt(
        research_question="How to improve training?",
        background_survey="Current methods use SGD...",
        candidates=test_candidates,
    )
    
    print("=" * 50)
    print("IR Probability Extractor - Prompt Test")
    print("=" * 50)
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Labels: {label_map}")
    print("\nUsage:")
    print("  extractor = IRProbabilityExtractor('http://localhost:30000/v1')")
    print("  result = extractor.get_selection_probabilities(...)")
