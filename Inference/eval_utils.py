"""
Shared utilities for evaluation scripts (hierarchical_search_eval, tournament_search_eval).
"""

import re
from typing import Tuple


# Keywords that typically introduce problem/limitation descriptions in background surveys
PROBLEM_KEYWORDS = [
    'however', 'limitation', 'gap', 'lack', 'missing', 'unclear', 
    'unknown', 'challenge', 'problem', 'issue', 'unresolved',
    'inadequate', 'insufficient', 'fail', 'cannot', 'unable'
]


def truncate_before_problem(
    text: str, 
    keywords: list = None
) -> Tuple[str, bool, str]:
    """
    Truncate background survey before the first problem/limitation description.
    
    This is used for ablation study to remove implicit motivation from background surveys,
    so that explicit motivation injection can be properly evaluated.
    
    Background surveys typically follow this structure:
        [Method introduction]  <-- Keep this part
        However, ...           <-- Truncate from here
        [Problem/limitation]   <-- Remove this (overlaps with motivation)
    
    Args:
        text: Original background survey text
        keywords: List of problem keywords to detect (default: PROBLEM_KEYWORDS)
    
    Returns:
        Tuple of:
        - truncated_text: The truncated text (or original if truncation not possible)
        - was_truncated: Whether truncation was actually applied
        - truncation_keyword: The keyword that triggered truncation (or empty string)
    """
    if not text:
        return text, False, ''
    
    if keywords is None:
        keywords = PROBLEM_KEYWORDS
    
    text_lower = text.lower()
    
    # Find the position of the first problem keyword
    first_pos = len(text)
    first_keyword = ''
    
    for kw in keywords:
        # Use word boundary to avoid partial matches
        match = re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
        if match and match.start() < first_pos:
            first_pos = match.start()
            first_keyword = kw
    
    # If no keyword found, return original text
    if first_pos == len(text):
        return text, False, ''
    
    # Find the sentence boundary before the keyword
    # Look for the last period/sentence-ending punctuation before the keyword
    truncate_pos = -1
    for punct in ['. ', '.\n', '.\t']:
        pos = text.rfind(punct, 0, first_pos)
        if pos > truncate_pos:
            truncate_pos = pos
    
    # If we found a valid sentence boundary, truncate there
    if truncate_pos > 0:
        truncated = text[:truncate_pos + 1]  # Include the period
        return truncated.strip(), True, first_keyword
    
    # No sentence boundary found - the first sentence already contains motivation
    # Return empty string; explicit motivation will provide the problem context
    return '', True, first_keyword


def parse_motivation_from_hypothesis_component(hypothesis_component: str) -> str:
    """
    Parse the detailed motivation from a hypothesis component string.
    
    Standard format (from prompt_store.py):
        Inspiration: [Key concept]
        - Motivation (WHY): [detailed motivation text]
        - Mechanism (HOW IT WORKS): [mechanism text]
        - Methodology (HOW IT'S INTEGRATED): [methodology text]
    
    Also handles variant formats like:
        #### 1. Motivation (WHY)
        [content]
    
    Returns:
        The motivation text, or empty string if section not found
    """
    if not hypothesis_component:
        return ''
    
    # Match various Motivation (WHY) section formats
    # - "- Motivation (WHY): content"
    # - "#### 1. Motivation (WHY)\ncontent"  
    # - "**Motivation (WHY):** content"
    pattern = r'(?:#+\s*\d*\.?\s*|[-*]\s*|\*{0,2})Motivation\s*\(WHY\)\s*\*{0,2}:?\s*(.*?)(?=\n(?:#+\s*\d*\.?\s*|[-*]\s*|\*{0,2})(?:Mechanism|Methodology)|$)'
    match = re.search(pattern, hypothesis_component, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    return ''
