import os, re, json, random, time, math, sys
import pandas as pd
from typing import List, Tuple, Dict
from openai import OpenAI, AzureOpenAI
import httpx

# Google GenAI is optional (only needed for api_type=2)
try:
    import google.genai as genai
    import google.genai.types as types
except ImportError:
    genai = None
    types = None


# =====================================================================
# Text extraction utilities
# =====================================================================

def extract_between_markers(source: str, label_regex: str):
    """Return text between '<label> starts' and '<label> ends'.

    Parameters
    ----------
    source : str
        The raw LLM response.
    label_regex : str
        A regex describing the label (e.g. ``'Research\\s*question'``).
        Should NOT contain ``starts``/``ends`` keywords.

    Returns
    -------
    str | None
        The extracted content, or None if the pattern is not found.
    """
    plain = re.sub(r'[\*_]+', '', source)
    pattern = rf'{label_regex}\s*starts\s*:?\s*([\s\S]+?)\s*{label_regex}\s*ends'
    m = re.search(pattern, plain, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    content = m.group(1).strip()
    return content if content else None


def extract_answer_content(text):
    """Extract content from DeepSeek-R1 format responses.

    Removes <think>...</think> tags and extracts the final answer.
    If no special tags found, returns the original text.
    """
    # Check for <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        text = answer_match.group(1).strip()
    else:
        # Handle content after </think>
        after_think_match = re.search(r'</think>\s*\n(.+)', text, re.DOTALL | re.IGNORECASE)
        if after_think_match:
            text = after_think_match.group(1).strip()
        else:
            # Remove <think> content
            patterns_to_remove = [
                r'<think>.*?</think>',
                r'<think>.*$',
                r'^.*</think>',
                r'</think>?\s*$',
                r'</think>',
                r'<think>',
            ]
            cleaned_text = text
            for pattern in patterns_to_remove:
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
            text = cleaned_text

    # Check for marked sections before aggressive cleanup
    has_marked_sections = bool(re.search(r'\w+\s+\d+\s+starts:', text, re.IGNORECASE))

    if not has_marked_sections:
        thinking_patterns = [
            r',\s*as\s+requested\.\s*$',
            r'\.\s*This\s+precisely.*$',
            r'\.\s*No\s+need\s+to\s+.*$',
            r'\.\s*I\'ll\s+.*$',
            r'\.\s*Let\s+me\s+.*$',
            r'\.\s*The\s+user\s+.*$',
        ]
        for pattern in thinking_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = text.strip()
        text = re.sub(r'\.\s+[a-z]+\s*$', '.', text)

    text = text.strip()
    return text if text else text


def init_llm_client(api_type, api_key, base_url, timeout=600.0):
    """Initialize LLM client based on API type.
    
    Args:
        api_type: 0 for OpenAI, 1 for Azure, 2 for Google
        api_key: API key for the service
        base_url: Base URL for the API endpoint
        timeout: Request timeout in seconds (default 120.0)
    
    Returns:
        Initialized client object
    """
    if api_type == 0:
        # OpenAI client - remove default 100 connection limit
        http_client = httpx.Client(
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
            timeout=httpx.Timeout(timeout)
        )
        return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    elif api_type == 1:
        # Azure client - remove default 100 connection limit
        http_client = httpx.Client(
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
            timeout=httpx.Timeout(timeout)
        )
        return AzureOpenAI(
            azure_endpoint=base_url,
            api_key=api_key,
            api_version="2024-06-01",
            http_client=http_client
        )
    elif api_type == 2:
        # Google client
        return genai.Client(api_key=api_key)
    else:
        raise NotImplementedError(f"API type {api_type} not supported")



# Call Openai API,k input is prompt, output is response
def llm_generation(prompt, model_name, client, temperature=1.0, api_type=0, if_filter_reasoning=None, max_tokens=None):
    # print("prompt: ", prompt)
    # adjust max_tokens if not explicitly provided
    if max_tokens is None:
        if "claude-3-haiku" in model_name:
            max_tokens = 4096
        else:
            max_tokens = 8192
    cnt_max_trials = 1
    # start inference util we get generation
    for cur_trial in range(cnt_max_trials):
        try:
            if api_type in [0, 1]:
                completion = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                    ]
                )
                generation = completion.choices[0].message.content.strip()
                
                # Simple truncation warning
                if completion.choices[0].finish_reason in ["length", "max_tokens"]:
                    print(f"⚠️ WARNING: Response truncated at {max_tokens} tokens. Consider increasing max_tokens.")
                
            # google client
            elif api_type == 2:
                response = client.models.generate_content(
                    model=model_name, 
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0)
                    )
                )
                generation = response.text.strip()
            else:
                raise NotImplementedError
            break
        except Exception as e:
            print("API Error occurred: ", e)
            time.sleep(0.25)
            if cur_trial == cnt_max_trials - 1:
                raise Exception("Failed to get generation after {} trials because of API error: {}.".format(cnt_max_trials, e))
    
    # Clean up R1/DeepSeek model's thinking tags if present
    # Use the imported extract_answer_content function which handles various formats
    # if if_filter_reasoning is not provided, set it based on model name
    if if_filter_reasoning is None:
        if "r1" in model_name.lower() or "deepseek" in model_name.lower():
            if_filter_reasoning = True
        else:
            if_filter_reasoning = False
    if if_filter_reasoning:
        generation = extract_answer_content(generation)
    
    # print("generation: ", generation)
    return generation



def extract_field(text, field_name, expected_type='text', strict_extraction=False):
    """Universal field extraction with type awareness.
    
    Args:
        text: The LLM response text
        field_name: The field to extract (e.g., "Hypothesis", "Answer", "Redundant")
        expected_type: 'text', 'bool'/'yes_no', 'number', etc.
        strict_extraction: If True, use stricter extraction for text fields to avoid contamination.
                          For text fields, strict mode will NEVER return the full text as fallback.
                          Recommended for extracting specific fields from structured responses.
    
    Returns:
        Extracted value in appropriate type, or None if extraction fails
    """
    # First try marker extraction (most reliable)
    result = extract_between_markers(text, field_name)
    
    # Process based on expected type
    if expected_type in ['bool', 'yes_no', 'boolean']:
        # For boolean/yes_no, check multiple sources
        if result:
            result_lower = result.lower().strip()
            if result_lower in ['yes', 'true', '1', 'correct', 'valid']:
                return True
            elif result_lower in ['no', 'false', '0', 'incorrect', 'invalid']:
                return False
        
        # Fallback: check beginning of response
        text_lower = text.lower().strip()
        if text_lower.startswith(('yes', 'true', 'correct')):
            return True
        if text_lower.startswith(('no', 'false', 'incorrect')):
            return False
        
        # Check first 100 characters
        first_part = text_lower[:100]
        yes_indicators = ['yes', 'true', 'correct', 'valid', 'sound', 'sufficient']
        no_indicators = ['no', 'false', 'incorrect', 'invalid', 'not', 'insufficient']
        
        yes_count = sum(1 for word in yes_indicators if word in first_part)
        no_count = sum(1 for word in no_indicators if word in first_part)
        
        if yes_count > no_count:
            return True
        elif no_count > yes_count:
            return False
        
        # Default based on context (conservative)
        return False
    
    elif expected_type == 'number':
        if result:
            # Extract number from result
            numbers = re.findall(r'\d+', result)
            if numbers:
                return int(numbers[0])
        # Fallback: search in text
        numbers = re.findall(r'\b(\d+)\b', text[:200])  # Check first 200 chars
        if numbers:
            return int(numbers[0])
        return None
    
    else:  # Default to text extraction
        if result:
            return result.strip()
        
        # Fallback: try pattern matching
        if strict_extraction:
            # Strict mode: More careful extraction to avoid contamination
            # Try multiple patterns in order of specificity
            
            # Escape field name for use in regex to handle special characters
            escaped_field = re.escape(field_name)
            
            patterns = [
                # Pattern 1: Field with "starts" and "ends" markers (highest priority)
                rf"{escaped_field}\s*starts\s*:?\s*(.+?)\s*{escaped_field}\s*ends",
                # Pattern 2: Markdown bold field pattern
                rf"\*\*{escaped_field}\*\*\s*:\s*([^\n]+)",
                # Pattern 3: Field with "is" pattern
                rf"{escaped_field}\s+is\s*:\s*([^\n]+)",
                # Pattern 4: Field with colon, stop at newline or next field
                rf"{escaped_field}\s*:\s*([^\n]+?)(?:\n|$)",
                # Pattern 5: More flexible pattern for edge cases
                rf"{escaped_field}\s*(?:starts\s*)?:\s*(.+?)(?:\n\n|\n(?=[A-Z][a-z]+\s*:)|\*\*[A-Z]|$)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    extracted = match.group(1).strip()
                    
                    # Clean up various markers and quotes
                    # Remove leading/trailing asterisks
                    extracted = re.sub(r'^\*+|\*+$', '', extracted).strip()
                    # Remove leading/trailing quotes (single or double)
                    extracted = re.sub(r'^["\']|["\']$', '', extracted).strip()
                    # Remove brackets if they wrap the entire string
                    if extracted.startswith('[') and extracted.endswith(']'):
                        extracted = extracted[1:-1].strip()
                    
                    # Remove any "ends" marker if present (only at the end)
                    ends_pattern = rf'\s*\*?\*?\s*{re.escape(field_name)}\s*ends?\s*\*?\*?\s*$'
                    extracted = re.sub(ends_pattern, '', extracted, flags=re.IGNORECASE).strip()
                    
                    # Sanity check: Check if we accidentally grabbed multiple fields
                    if extracted:
                        # Generic check: Look for patterns that indicate field boundaries
                        # Pattern 1: "\n[FieldName]:" or "\n**[FieldName]" (newline + capitalized word + colon/marker)
                        # Pattern 2: Multiple "starts/ends" markers
                        
                        # Check for other field-like patterns (newline followed by Capitalized word and colon)
                        # This catches patterns like "\nAnswer:", "\nHypothesis:", etc. without hardcoding them
                        field_pattern = r'\n\s*(?:\*\*)?[A-Z][a-zA-Z\s]+(?:\*\*)?\s*:'
                        if re.search(field_pattern, extracted):
                            continue  # Try next pattern, we likely grabbed too much
                        
                        # Check for multiple "starts/ends" markers which indicate over-extraction
                        starts_ends_pattern = r'[A-Z][a-zA-Z\s]+\s+(?:starts\s*:|ends)'
                        if re.search(starts_ends_pattern, extracted, re.IGNORECASE):
                            continue  # Try next pattern, we likely grabbed too much
                        
                        # Otherwise accept the extraction
                        return extracted
            
            # In strict mode, don't return anything else
            return None
        else:
            # Non-strict mode with safer fallback
            # Escape field name for safety
            escaped_field = re.escape(field_name)
            
            # Try basic pattern first
            pattern = rf"{escaped_field}[:\s]+(.+?)(?:\n|$)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                # Clean up common markers
                extracted = re.sub(r'^\*+|\*+$', '', extracted).strip()
                extracted = re.sub(r'^["\']|["\']$', '', extracted).strip()
                
                # Safety check: reject if we grabbed multiple fields
                if extracted:
                    # Generic check for field boundaries (same as strict mode)
                    field_pattern = r'\n\s*(?:\*\*)?[A-Z][a-zA-Z\s]+(?:\*\*)?\s*:'
                    starts_ends_pattern = r'[A-Z][a-zA-Z\s]+\s+(?:starts\s*:|ends)'
                    
                    if not re.search(field_pattern, extracted) and \
                       not re.search(starts_ends_pattern, extracted, re.IGNORECASE):
                        return extracted
            
            # DEPRECATED: Full text fallback - kept for backward compatibility but with warning
            # This is dangerous and should be avoided in new code
            if 50 < len(text) < 2000:
                # Log warning if possible (check if print is acceptable in this context)
                import sys
                print(f"WARNING: extract_field() returning full text for field '{field_name}'. "
                      f"Consider using strict_extraction=True to avoid this.", file=sys.stderr)
                return text.strip()
            
            return None

def simple_retry_on_429(func, *args, max_retries=10, initial_delay=1, max_wait=30, **kwargs):
    """Retry function calls on 429 errors with exponential backoff
    
    This is a standalone retry function with no dependencies.
    Optimized for API rate limits (NCBI, Semantic Scholar).
    
    Args:
        func: Function to call
        max_retries: Maximum number of retry attempts (default 10)
        initial_delay: Initial delay in seconds (default 1)
        max_wait: Maximum wait time per retry in seconds (default 30)
        *args, **kwargs: Arguments to pass to func
    """
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            # Check if it's a retryable error
            is_retryable = (
                '429' in error_str or 'rate' in error_str or 'too many' in error_str or
                '503' in error_str or 'service unavailable' in error_str or
                'connection' in error_str or  # Catches "connection error", "connectionrefusederror", etc.
                error_type == 'RetryError'  # semanticscholar wrapper
            )
            
            if is_retryable and i < max_retries - 1:
                # Exponential backoff with cap
                delay = min(initial_delay * (2 ** i), max_wait) + random.uniform(0, 1)
                print(f"  Rate limited (attempt {i+1}/{max_retries}), waiting {delay:.1f}s...")
                time.sleep(delay)
                continue
            # Re-raise if not retryable or last attempt
            raise
    return None


def jaccard_similarity(str1: str, str2: str) -> float:
    """
    Calculate Jaccard similarity between two strings based on word overlap.
    The score range is [0, 1].
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Jaccard similarity score in range [0, 1]
    """
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
        
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def match_output_to_exact_candidate(
    output: str, 
    candidates: List[str], 
    if_print_warning: bool = True
) -> Tuple[str, float]:
    """
    Match a generated/extracted output to the most similar candidate from a list.
    Useful when LLM-generated outputs have slight differences from exact candidates.
    
    Args:
        output: Output to match (e.g., from LLM generation)
        candidates: List of candidate strings to match against
        if_print_warning: Whether to print warning for low similarity matches
        
    Returns:
        Tuple of (best_matching_candidate, similarity_score)
    """
    # Clean the output
    output = output.strip().strip('"').strip("'").strip()
    
    # Calculate similarity with each candidate
    similarities = []
    for candidate in candidates:
        similarity = jaccard_similarity(output.lower(), candidate.lower())
        similarities.append(similarity)
    
    # Get the best match
    max_similarity = max(similarities)
    best_match_idx = similarities.index(max_similarity)
    best_match = candidates[best_match_idx]
    
    # Warning for poor matches
    if max_similarity < 0.3 and if_print_warning:
        print(f"Warning: Low similarity match (score: {max_similarity:.3f})")
        print(f"  Original: {output}")
        print(f"  Matched to: {best_match}\n")
    
    return best_match, max_similarity


# only consider exact match
def clean_eos_tokens(text: str) -> str:
    """
    Strip all existing EOS tokens from the end of text.
    This ensures clean text for the training framework to process.
    The training framework will add appropriate tokens based on its template setting.
    
    Args:
        text: The text to clean EOS tokens from
        
    Returns:
        The text with all EOS tokens removed from the end
        
    Example:
        >>> text = "This is a response<｜end▁of▁sentence｜><|im_end|>"
        >>> clean_eos_tokens(text)
        "This is a response"
    """
    if not text:
        return text
    
    # Common EOS tokens used by different models
    eos_tokens_to_strip = [
        '<|im_end|>',              # ChatML format
        '<｜end▁of▁sentence｜>',    # DeepSeek-R1/Qwen distilled models  
        '</s>',                    # Llama/Mistral models
        '<|endoftext|>',           # GPT models
        '[/INST]',                 # Some instruction-tuned models
        '<|eot_id|>',             # Some newer models
    ]
    
    # Keep stripping until no more tokens are found at the end
    # This handles cases with multiple or nested EOS tokens
    changed = True
    while changed:
        changed = False
        for token in eos_tokens_to_strip:
            if text.endswith(token):
                text = text[:-len(token)]
                changed = True
                break  # Start over to maintain order
    
    return text

def calculate_retrieval_accuracy(
    predictions: List[str],
    ground_truth: str
) -> Dict:
    """
    Calculate accuracy metrics for any retrieval/selection task (only exact match is considered).
    
    Args:
        predictions: List of predictions from multiple samples
        ground_truth: The correct answer
        
    Returns:
        Dictionary with accuracy metrics:
        - accuracy: Percentage of correct predictions
        - correct_count: Number of correct predictions  
        - total_count: Total number of predictions
    """
    if not predictions:
        return {
            'accuracy': 0.0,
            'correct_count': 0,
            'total_count': 0
        }
    
    # Count correct predictions
    correct_count = sum(1 for pred in predictions if pred == ground_truth)
    total_count = len(predictions)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': total_count
    }

def extract_hypothesis_from_response(raw_response: str) -> tuple:
    """
    Extract hypothesis and reasoning trace from model response.
    
    Designed for DeepSeek-R1 style outputs with <think>...</think> tags.
    Handles edge cases like prompt templates containing </think> examples.
    
    Strategy:
    1. Find the LAST </think> (model's actual output, not prompt template)
    2. Extract reasoning from the last <think>...</think> pair
    3. Extract hypothesis from content AFTER the last </think>
    
    Args:
        raw_response: Raw model output string
        
    Returns:
        Tuple of (generated_hypothesis, reasoning_trace)
        Returns (None, None) if input is empty
    """
    if not raw_response or not raw_response.strip():
        return None, None
    
    # Step 1: Find the LAST </think> (model's actual output, not prompt template)
    think_end_pos = raw_response.rfind('</think>')
    
    # Step 2: Extract reasoning
    # Find the <think> that corresponds to the last </think>
    reasoning_trace = None
    if think_end_pos != -1:
        # Look for the last <think> before the last </think>
        think_start_pos = raw_response.rfind('<think>', 0, think_end_pos)
        if think_start_pos != -1:
            reasoning_trace = raw_response[think_start_pos + 7:think_end_pos].strip()
        else:
            reasoning_trace = raw_response[:think_end_pos].strip()
    else:
        reasoning_trace = raw_response.strip()
    
    # Step 3: Extract hypothesis from content AFTER the last </think>
    # This ensures we get the model's actual output, not prompt template
    gene_hyp = None
    if think_end_pos != -1:
        content_after_think = raw_response[think_end_pos + len('</think>'):]
        # Try markers within the content after </think>
        gene_hyp = extract_between_markers(content_after_think, r'Delta\s*Hypothesis')
        # Fallback: use all content after </think>
        if not gene_hyp:
            gene_hyp = content_after_think.strip()
    else:
        # No </think> found - try markers on full response
        gene_hyp = extract_between_markers(raw_response, r'Delta\s*Hypothesis')
        if not gene_hyp:
            gene_hyp = raw_response.strip()
    
    return gene_hyp, reasoning_trace

