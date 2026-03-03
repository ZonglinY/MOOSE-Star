import os
import sys
import json
import argparse
import time
import tempfile
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
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
    match_output_to_exact_candidate,
    extract_field
)

# Import prompt_store for generating prompts from raw data
from prompt_store import instruction_prompts

# Alphabetical labels for candidates (matches SFT data preparation)
LABELS = [chr(ord('A') + i) for i in range(26)]


def convert_raw_sample_to_eval_format(raw_sample: List, if_shuffle_candidates: bool = True) -> Dict:
    """
    Convert raw data format to evaluation format with conversations structure.
    
    Raw data format (from collected_inspiration_retrieval_QA_data_*.json):
    [background, negative_inspirations, ground_truth, year_pmid]
    
    Where:
    - background: [research_question, survey, pre_step_hyp]
    - negative_inspirations: [[title, abstract, year], ...] (list of negative papers)
    - ground_truth: [title, abstract, inspiration, relation]
    - year_pmid: "YYYY_PMID" string
    
    Args:
        raw_sample: Raw data sample in list format (4 elements)
        if_shuffle_candidates: Whether to shuffle candidate order (default True)
        
    Returns:
        Dict with 'conversations' field matching the evaluation format
    """
    # Parse raw sample (always 4 elements)
    if len(raw_sample) != 4:
        raise ValueError(f"Expected 4-element sample, got {len(raw_sample)} elements")
    
    background = raw_sample[0]  # [research_question, survey, pre_step_hyp]
    negative_inspirations = raw_sample[1]  # [[title, abstract, year], ...]
    ground_truth = raw_sample[2]  # [title, abstract, inspiration, relation]
    year_pmid = raw_sample[3]  # "YYYY_PMID" string
    
    research_question = background[0]
    background_survey = background[1]
    pre_step_hyp = background[2]  # Can be None
    
    gdth_title = ground_truth[0]
    gdth_abstract = ground_truth[1]
    
    # Prepare all candidates (ground truth + negatives)
    all_candidates = []
    
    # Add ground truth
    all_candidates.append({
        'title': gdth_title,
        'abstract': gdth_abstract
    })
    
    # Add negatives
    for neg_insp in negative_inspirations:
        all_candidates.append({
            'title': neg_insp[0],
            'abstract': neg_insp[1]
        })
    
    # Shuffle candidates to avoid position bias
    if if_shuffle_candidates:
        indices = list(range(len(all_candidates)))
        random.shuffle(indices)
        shuffled_candidates = [all_candidates[i] for i in indices]
    else:
        shuffled_candidates = all_candidates
    
    # Create prompt using prompt_store (same as rejection sampling)
    prompts = instruction_prompts("inspiration_retrieval_with_reasoning_with_alphabetical_candidates")
    
    # Format previous hypothesis section
    if pre_step_hyp:
        pre_hyp_text = pre_step_hyp
    else:
        pre_hyp_text = "None (starting from background knowledge)"
    
    # Format candidates section with alphabetical labels (matches SFT data)
    candidates_list = []
    for i, candidate in enumerate(shuffled_candidates):
        label = LABELS[i]  # A, B, C, ...
        candidates_list.append(f"""### Candidate [{label}]
**Title:** {candidate['title']}
**Abstract:** {candidate['abstract']}""")
    
    candidates_section = "\n\n".join(candidates_list)
    
    # Build the user prompt (same structure as rejection sampling)
    user_prompt = (prompts[0] + research_question + 
                   prompts[1] + background_survey + 
                   prompts[2] + pre_hyp_text + 
                   prompts[3] + candidates_section + 
                   prompts[4])
    
    # R1-Distill Native Format: No system prompt (user prompt already contains task instructions)
    # Create conversations structure (only user needed for evaluation)
    # Note: We don't need expected_response in conversations since we use ground_truth_title directly
    result = {
        'conversations': [
            {'role': 'user', 'content': user_prompt}
        ]
    }
    
    # Preserve metadata for evaluation and debugging
    # candidate_titles in the same order as they appear in the prompt (after shuffling)
    result['candidate_titles'] = [c['title'] for c in shuffled_candidates]
    result['ground_truth_title'] = gdth_title
    result['num_candidates'] = len(all_candidates)
    result['year_pmid'] = year_pmid
    
    # Map labels to titles for ID-based extraction
    # e.g., {'A': 'Title1', 'B': 'Title2', ...}
    result['label_to_title'] = {LABELS[i]: c['title'] for i, c in enumerate(shuffled_candidates)}
    
    # Find the label for ground truth title
    for i, c in enumerate(shuffled_candidates):
        if c['title'] == gdth_title:
            result['ground_truth_label'] = LABELS[i]
            break
    
    return result


def get_overlapping_year_pmids(overlapping_dir: str) -> set:
    """
    Get set of year_pmid values from overlapping directory.
    
    The overlapping directory contains files named like "2025_40610703.json"
    where the filename (without .json) is the year_pmid to filter out.
    
    Args:
        overlapping_dir: Path to directory containing overlapping sample files
        
    Returns:
        Set of year_pmid strings to filter out
    """
    import os
    overlapping_pmids = set()
    
    if not os.path.isdir(overlapping_dir):
        print(f"Warning: Overlapping directory not found: {overlapping_dir}")
        return overlapping_pmids
    
    for filename in os.listdir(overlapping_dir):
        if filename.endswith('.json'):
            year_pmid = filename.replace('.json', '')
            overlapping_pmids.add(year_pmid)
    
    return overlapping_pmids


def load_raw_data(
    data_path: str, 
    if_shuffle_candidates: bool = True, 
    random_seed: int = None,
    overlapping_dir: str = None
) -> List[Dict]:
    """
    Load raw data and convert to evaluation format.
    
    Args:
        data_path: Path to raw JSON data file
        if_shuffle_candidates: Whether to shuffle candidate order
        random_seed: Random seed for reproducible shuffling (None for random)
        overlapping_dir: Path to directory with overlapping samples to filter out
        
    Returns:
        List of samples in evaluation format (with conversations structure)
    """
    print(f"Loading raw data from {data_path}")
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    original_count = len(raw_data)
    
    # Filter out overlapping samples if directory provided
    if overlapping_dir:
        overlapping_pmids = get_overlapping_year_pmids(overlapping_dir)
        if overlapping_pmids:
            raw_data = [s for s in raw_data if s[3] not in overlapping_pmids]
            filtered_count = original_count - len(raw_data)
            print(f"Filtered out {filtered_count} samples overlapping with training data")
    
    # Set random seed for reproducible shuffling
    if random_seed is not None:
        random.seed(random_seed)
        print(f"Using random seed: {random_seed} for candidate shuffling")
    
    print(f"Converting {len(raw_data)} raw samples to evaluation format...")
    
    eval_samples = []
    for raw_sample in tqdm(raw_data, desc="Converting samples"):
        eval_sample = convert_raw_sample_to_eval_format(raw_sample, if_shuffle_candidates)
        eval_samples.append(eval_sample)
    
    print(f"Converted {len(eval_samples)} samples")
    return eval_samples


class InspirationRetrievalEvaluator:
    """
    Evaluator for inspiration retrieval models.
    Tests accuracy on selecting the correct inspiration from candidates.
    
    Performance optimizations (borrowed from hypothesis_composition_eval_rubric.py):
    - Flash Attention 2 for 2x faster attention computation
    - Batch generation for better GPU utilization
    - torch.inference_mode() for more efficient inference
    - KV cache enabled for faster generation
    """
    
    def __init__(
        self,
        model_path: str,
        lora_path: str = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
        max_length: int = 16384,
        max_new_tokens: int = 4096,
        max_retries: int = 2,
        similarity_threshold: float = 0.3,
        debug_template: bool = False,
        batch_size: int = 1,
        temperature: float = 0.1,
        top_p: float = 0.95
    ):
        """
        Initialize the evaluator with a model.
        
        Args:
            model_path: Path to base model
            lora_path: Path to LoRA checkpoint (None for base model evaluation)
            device: Device to use (cuda/cpu)
            load_in_8bit: Whether to load model in 8-bit precision
            max_length: Maximum sequence length
            max_new_tokens: Maximum new tokens to generate
            max_retries: Maximum retry attempts for poor extractions (default: 2, reduced for efficiency)
            similarity_threshold: Minimum acceptable similarity (default: 0.3, matches SFT data minimum)
            debug_template: Whether to print template information for debugging
            batch_size: Batch size for generation (higher = better GPU utilization)
            temperature: Generation temperature
            top_p: Top-p sampling parameter
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        self.max_retries = max_retries
        self.similarity_threshold = similarity_threshold
        self.debug_template = debug_template
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p
        
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
            
            # First attempt: try loading normally
            try:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    lora_path,
                    torch_dtype=torch.bfloat16
                )
            except (TypeError, AttributeError) as e:
                error_str = str(e)
                
                # Handle unsupported config parameters
                if "unexpected keyword argument" in error_str:
                    print(f"Detected compatibility issue: {error_str}")
                    print("Attempting to fix by removing unsupported parameters...")
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Copy all files and clean adapter_config.json
                        for item in Path(lora_path).iterdir():
                            if item.name == "adapter_config.json":
                                with open(item, 'r') as f:
                                    config = json.load(f)
                                
                                # Remove ALL non-standard LoRA parameters
                                # Keep only standard LoRA config parameters
                                standard_lora_params = {
                                    'r', 'lora_alpha', 'lora_dropout', 'target_modules',
                                    'bias', 'task_type', 'inference_mode', 'modules_to_save',
                                    'peft_type', 'auto_mapping', 'base_model_name_or_path',
                                    'revision', 'layers_to_transform', 'layers_pattern',
                                    'rank_pattern', 'alpha_pattern', 'use_rslora',
                                    'use_dora', 'loftq_config', 'megatron_config',
                                    'fan_in_fan_out', 'layer_replication', 'runtime_config'
                                }
                                
                                # Find and remove non-standard parameters
                                params_to_remove = [k for k in config.keys() if k not in standard_lora_params]
                                
                                if params_to_remove:
                                    print(f"  Removing non-standard parameters: {params_to_remove}")
                                    for param in params_to_remove:
                                        config.pop(param, None)
                                
                                with open(Path(temp_dir) / item.name, 'w') as f:
                                    json.dump(config, f, indent=2)
                            else:
                                shutil.copy2(item, Path(temp_dir) / item.name)
                        
                        # Try loading with cleaned config
                        try:
                            self.model = PeftModel.from_pretrained(
                                self.model,
                                temp_dir,
                                torch_dtype=torch.bfloat16
                            )
                            print("Successfully loaded LoRA with cleaned config")
                        except AttributeError as ae:
                            if "memory_efficient_backward" in str(ae) and load_in_8bit:
                                print("Encountered 8-bit quantization compatibility issue.")
                                print("Reloading model in bfloat16 instead of 8-bit...")
                                
                                # Reload base model without 8-bit quantization
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    model_path,
                                    torch_dtype=torch.bfloat16,
                                    device_map="auto",
                                    trust_remote_code=True
                                )
                                
                                # Now load LoRA
                                self.model = PeftModel.from_pretrained(
                                    self.model,
                                    temp_dir,
                                    torch_dtype=torch.bfloat16
                                )
                                print("Successfully loaded LoRA in bfloat16 precision")
                            else:
                                raise
                
                # Handle 8-bit quantization issues directly
                elif "memory_efficient_backward" in error_str and load_in_8bit:
                    print("Detected 8-bit quantization compatibility issue.")
                    print("Reloading model in bfloat16 instead of 8-bit...")
                    
                    # Reload base model without 8-bit quantization
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    
                    # Try loading LoRA again
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        lora_path,
                        torch_dtype=torch.bfloat16
                    )
                    print("Successfully loaded LoRA in bfloat16 precision")
                else:
                    raise  # Re-raise if it's a different type of error
            
            # Merge LoRA weights for faster inference
            self.model = self.model.merge_and_unload()
            print("LoRA weights loaded and merged")
        
        self.model.eval()
        print(f"Model loaded successfully (batch_size={self.batch_size})")
        
        # Debug template information if requested
        if self.debug_template:
            print("\n=== Template Debug Information ===")
            if hasattr(self.tokenizer, 'chat_template'):
                if self.tokenizer.chat_template:
                    print(f"Chat template found: {self.tokenizer.chat_template[:200]}...")
                    # Test the template - R1-Distill Native Format (no system prompt)
                    test_messages = [
                        {"role": "user", "content": "Test user"}
                    ]
                    # Show the format we use (add_generation_prompt=False + manual <｜Assistant｜>)
                    test_prompt = self.tokenizer.apply_chat_template(
                        test_messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    test_prompt += "<｜Assistant｜>"
                    print(f"R1-Distill Native Format (manual <｜Assistant｜>, model generates <think>):\n{test_prompt}")
                    print("\nNote: Model generates <think>\\n as first token")
                else:
                    print("Chat template attribute exists but is None")
            else:
                print("No chat template attribute found")
            print("=================================\n")
    
    def generate_response(self, prompt: str, temperature: float = None) -> str:
        """
        Generate a response from the model (single prompt).
        
        Args:
            prompt: Input prompt
            temperature: Generation temperature (lower = more deterministic)
            
        Returns:
            Generated response text
        """
        if temperature is None:
            temperature = self.temperature
            
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        ).to(self.device)
        
        # Generate with inference_mode for better performance
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=self.top_p,
                num_beams=1,
                use_cache=True,  # Enable KV cache
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response
    
    def generate_responses_batch(self, prompts: List[str], temperature: float = None) -> List[str]:
        """
        Generate responses for multiple prompts in a batch for better GPU utilization.
        
        Args:
            prompts: List of formatted prompts (already includes chat template)
            temperature: Generation temperature
            
        Returns:
            List of generated response texts
        """
        if not prompts:
            return []
        
        if temperature is None:
            temperature = self.temperature
        
        # Tokenize with left padding for batch generation
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        inputs = self.tokenizer(
            prompts,
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
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=self.top_p,
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
    
    
    def extract_selected_id(self, response: str, label_to_title: Dict[str, str]) -> Tuple[str, str, str, bool]:
        """
        Extract the selected ID from model response and map to candidate title.
        
        The model outputs format: **Selected ID starts:** [X] **Selected ID ends**
        where X is a letter (A-Z) corresponding to a candidate.
        
        Args:
            response: Model's response
            label_to_title: Mapping from label (A, B, C, ...) to candidate title
            
        Returns:
            Tuple of (raw_extracted_id, matched_label, matched_title, should_retry)
        """
        import re
        
        # Try to extract the Selected ID field
        # Pattern: **Selected ID starts:** [X] **Selected ID ends**
        selected_id = extract_field(response, "Selected ID", 
                                   expected_type='text', strict_extraction=True)
        
        if not selected_id:
            # Fallback: try to find [X] pattern near the end of response
            # Look for patterns like "[A]", "[B]", etc.
            matches = re.findall(r'\[([A-Z])\]', response[-500:] if len(response) > 500 else response)
            if matches:
                selected_id = f"[{matches[-1]}]"  # Use the last match
            else:
                return "", "", "", True  # Retry if no ID extracted
        
        # Clean up and extract the letter
        selected_id = selected_id.strip()
        
        # Extract the letter from formats like "[A]", "A", "[A", "A]"
        match = re.search(r'\[?([A-Z])\]?', selected_id)
        if not match:
            return selected_id, "", "", True  # Retry if no valid letter found
        
        label = match.group(1)  # The letter (A, B, C, ...)
        
        # Map label to title
        if label in label_to_title:
            matched_title = label_to_title[label]
            return selected_id, label, matched_title, False
        else:
            # Label not in valid range
            return selected_id, label, "", True
    
    def extract_selected_title(self, response: str, candidate_titles: List[str]) -> Tuple[str, str, bool]:
        """
        Legacy method: Extract the selected title from model response and match to candidates.
        
        This is kept for backward compatibility with models that output full titles.
        Uses similarity threshold of 0.3 (matching SFT data minimum) to determine if retry is needed.
        
        Args:
            response: Model's response
            candidate_titles: List of candidate titles
            
        Returns:
            Tuple of (raw_extracted_title, matched_candidate_title, should_retry)
        """
        # Use strict extraction to avoid getting the full response
        selected_title = extract_field(response, "Selected Title", 
                                      expected_type='text', strict_extraction=True)
        
        if not selected_title:
            return "", "", True  # Retry if no title extracted
        
        # Clean up the extracted title
        # Note: extract_field does basic cleanup, but we do extra cleaning for edge cases
        # like nested brackets [["Title"]] or multiple quote types
        selected_title = selected_title.strip().strip('"\'*[]')
        
        # Check if title became empty after cleaning
        if not selected_title:
            return "", "", True  # Retry if title is empty after cleaning
        
        # Match with candidates
        matched_title, similarity = match_output_to_exact_candidate(
            selected_title, 
            candidate_titles,
            if_print_warning=False
        )
        
        # Retry only if similarity is below acceptable threshold (0.3)
        # This matches the effective minimum in SFT data preparation
        should_retry = similarity < self.similarity_threshold
        
        if should_retry:
            return selected_title, "", True
        
        return selected_title, matched_title, False
    
    # Input:
    #   - sample: {
    #       'conversations': [
    #           {'role': 'system', 'content': 'system prompt text'},
    #           {'role': 'user', 'content': 'user prompt with candidate titles'},
    #           {'role': 'assistant', 'content': 'expected response with selected title'}
    #       ]
    #   }
    def evaluate_single_sample(
        self,
        sample: Dict,
        temperature: float = 0.1,
        verbose: bool = False
    ) -> Dict:
        """
        Evaluate a single sample.
        
        Args:
            sample: Sample with conversations (system, user, assistant)
            temperature: Generation temperature
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary with evaluation results
        """
        # Extract components from sample
        conversations = sample['conversations']
        # R1-Distill Native Format: No system prompt, only user message
        user_prompt = conversations[0]['content']
        
        # Get ground truth and candidate metadata
        # (set by convert_raw_sample_to_eval_format)
        ground_truth_title = sample.get('ground_truth_title')
        ground_truth_label = sample.get('ground_truth_label')
        candidate_titles = sample.get('candidate_titles')
        label_to_title = sample.get('label_to_title')
        
        if not ground_truth_title:
            raise ValueError("Sample missing 'ground_truth_title' metadata. "
                           "Ensure data was converted with convert_raw_sample_to_eval_format.")
        if not candidate_titles:
            raise ValueError("Sample missing 'candidate_titles' metadata. "
                           "Ensure data was converted with convert_raw_sample_to_eval_format.")
        if not label_to_title:
            raise ValueError("Sample missing 'label_to_title' metadata. "
                           "Ensure data was converted with convert_raw_sample_to_eval_format.")
        if not ground_truth_label:
            raise ValueError("Sample missing 'ground_truth_label' metadata. "
                           "Ensure data was converted with convert_raw_sample_to_eval_format.")
        
        # Sanity check: ground truth should be among candidates
        if ground_truth_title not in candidate_titles:
            raise ValueError(f"Ground truth title not found among candidates. "
                           f"This indicates a data conversion issue.")
        
        # R1-Distill Native Format:
        # - No system prompt (user prompt already contains task instructions)
        # - Training data includes <think>\n at start of assistant content
        # - Use add_generation_prompt=False and manually add <｜Assistant｜>
        # - Model generates: <think>\n[reasoning]\n</think>\n\n**Selected Title...**
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        
        # Use add_generation_prompt=False to avoid adding <think>\n to prompt
        # (model learned to generate <think>\n as first token)
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            # Manually add <｜Assistant｜> - model will generate <think>\n itself
            full_prompt += "<｜Assistant｜>"
        else:
            # DeepSeek-R1-Distill models should always have a chat template
            # Fail explicitly rather than using wrong format
            raise ValueError(
                "No chat template found. DeepSeek-R1-Distill models should have a chat template defined. "
                "Check that the model was loaded correctly from the expected path."
            )
        
        # Generate response with minimal retry logic for efficiency
        retry_count = 0
        total_generation_time = 0
        # No need to initialize loop variables - max_retries is validated to be >= 0
        
        while retry_count <= self.max_retries:
            if verbose and retry_count > 0:
                print(f"  Retry {retry_count}/{self.max_retries} due to extraction failure")
            
            start_time = time.time()
            generated_response = self.generate_response(
                full_prompt, 
                temperature * (1 + retry_count * 0.1)  # Increase temperature on retry
            )
            total_generation_time += time.time() - start_time
            
            # Extract selected ID (new format: [A], [B], etc.)
            selected_id_raw, selected_label, selected_title_matched, should_retry = self.extract_selected_id(
                generated_response, 
                label_to_title
            )
            
            # Stop if extraction succeeds or max retries reached
            if not should_retry or retry_count >= self.max_retries:
                break
            
            retry_count += 1
        
        # Check correctness by comparing model's selected label with ground truth label
        is_correct = bool(selected_label and selected_label == ground_truth_label)
        
        result = {
            'generated_response': generated_response,
            'selected_id_raw': selected_id_raw,
            'selected_label': selected_label,
            'selected_title': selected_title_matched,
            'expected_label': ground_truth_label,
            'expected_title': ground_truth_title,
            'candidate_titles': candidate_titles,
            'num_candidate_titles': len(candidate_titles),
            'is_correct': is_correct,
            'generation_time': total_generation_time,
            'retry_count': retry_count
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Selected: [{selected_label}] {selected_title_matched}")
            print(f"Expected: [{ground_truth_label}] {ground_truth_title}")
            print(f"Correct: {is_correct}")
            print(f"Time: {total_generation_time:.2f}s")
            if retry_count > 0:
                print(f"Retries: {retry_count}")
        
        return result
    
    def _build_full_prompt(self, sample: Dict) -> str:
        """
        Build the full prompt from a sample (for batch generation).
        
        Args:
            sample: Sample with conversations structure
            
        Returns:
            Full formatted prompt string
        """
        conversations = sample['conversations']
        user_prompt = conversations[0]['content']
        
        messages = [{"role": "user", "content": user_prompt}]
        
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            full_prompt += "<｜Assistant｜>"
        else:
            raise ValueError("No chat template found. DeepSeek-R1-Distill models should have a chat template.")
        
        return full_prompt
    
    def _process_batch_response(
        self, 
        sample: Dict, 
        response: str
    ) -> Dict:
        """
        Process a single response from batch generation.
        
        Args:
            sample: Original sample
            response: Generated response
            
        Returns:
            Result dictionary
        """
        result, _ = self._process_batch_response_with_retry_flag(sample, response)
        return result
    
    def _process_batch_response_with_retry_flag(
        self, 
        sample: Dict, 
        response: str
    ) -> Tuple[Dict, bool]:
        """
        Process a single response from batch generation, with retry flag.
        
        Args:
            sample: Original sample
            response: Generated response
            
        Returns:
            Tuple of (result dictionary, should_retry flag)
        """
        ground_truth_title = sample.get('ground_truth_title')
        ground_truth_label = sample.get('ground_truth_label')
        candidate_titles = sample.get('candidate_titles')
        label_to_title = sample.get('label_to_title')
        
        # Extract selected ID
        selected_id_raw, selected_label, selected_title_matched, should_retry = self.extract_selected_id(
            response, label_to_title
        )
        
        is_correct = bool(selected_label and selected_label == ground_truth_label)
        
        result = {
            'generated_response': response,
            'selected_id_raw': selected_id_raw,
            'selected_label': selected_label,
            'selected_title': selected_title_matched,
            'expected_label': ground_truth_label,
            'expected_title': ground_truth_title,
            'candidate_titles': candidate_titles,
            'num_candidate_titles': len(candidate_titles) if candidate_titles else 0,
            'is_correct': is_correct,
            'generation_time': 0,  # Batch timing not per-sample
            'retry_count': 0
        }
        
        return result, should_retry
    
    def evaluate_dataset(
        self,
        data_file_path: str,
        output_path: str = None,
        max_samples: int = None,
        temperature: float = None,
        verbose: bool = False,
        if_shuffle_candidates: bool = True,
        random_seed: int = None,
        overlapping_dir: str = None,
        start_idx: int = None,
        end_idx: int = None
    ) -> Dict:
        """
        Evaluate the model on a dataset with batch processing for better GPU utilization.
        
        Args:
            data_file_path: Path to raw data JSON file (collected_inspiration_retrieval_QA_data_*.json)
            output_path: Path to save results (optional)
            max_samples: Maximum number of samples to evaluate (None for all)
            temperature: Generation temperature (None uses self.temperature)
            verbose: Whether to print detailed output
            if_shuffle_candidates: Whether to shuffle candidate order
            random_seed: Random seed for reproducible shuffling
            overlapping_dir: Path to directory with samples to filter out (train/eval overlap)
            start_idx: Start index for parallel processing (0-based, inclusive)
            end_idx: End index for parallel processing (exclusive)
            
        Returns:
            Dictionary with overall evaluation metrics
        """
        if temperature is None:
            temperature = self.temperature
            
        # Load raw data and convert to evaluation format
        samples = load_raw_data(data_file_path, if_shuffle_candidates, random_seed, overlapping_dir)
        
        if not samples:
            raise ValueError(f"No samples found in {data_file_path}. Check if file is empty.")
        
        # Apply index range for parallel processing
        if start_idx is not None or end_idx is not None:
            start = start_idx if start_idx is not None else 0
            end = end_idx if end_idx is not None else len(samples)
            samples = samples[start:end]
            print(f"Processing samples [{start}:{end}] ({len(samples)} samples)")
        
        if max_samples:
            samples = samples[:max_samples]
        
        total_samples = len(samples)
        print(f"Evaluating {total_samples} samples (batch_size={self.batch_size})")
        
        # Build all prompts first
        all_prompts = []
        for sample in samples:
            try:
                prompt = self._build_full_prompt(sample)
                all_prompts.append(prompt)
            except Exception as e:
                print(f"Error building prompt: {e}")
                all_prompts.append(None)
        
        # Process in batches - Phase 1: Initial generation (no retries yet)
        # Use list to store (result, should_retry) tuples, indexed by global sample index
        all_results_with_retry_flags = [None] * total_samples
        total_time = 0
        
        pbar = tqdm(total=total_samples, desc="Evaluating")
        
        for batch_start in range(0, total_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_samples)
            batch_samples = samples[batch_start:batch_end]
            batch_prompts = all_prompts[batch_start:batch_end]
            
            # Filter out None prompts (use local indices within batch)
            valid_local_indices = [i for i, p in enumerate(batch_prompts) if p is not None]
            valid_prompts = [batch_prompts[i] for i in valid_local_indices]
            
            if not valid_prompts:
                # All prompts in batch failed
                for i in range(len(batch_samples)):
                    global_idx = batch_start + i
                    result = {'error': 'Failed to build prompt', 'is_correct': False, 'retry_count': 0}
                    all_results_with_retry_flags[global_idx] = (result, False)
                pbar.update(len(batch_samples))
                continue
            
            # Generate batch responses
            start_time = time.time()
            try:
                batch_responses = self.generate_responses_batch(valid_prompts, temperature)
            except Exception as e:
                print(f"Error in batch generation: {e}")
                for i in range(len(batch_samples)):
                    global_idx = batch_start + i
                    result = {'error': str(e), 'is_correct': False, 'retry_count': 0}
                    all_results_with_retry_flags[global_idx] = (result, False)
                pbar.update(len(batch_samples))
                continue
            
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # Process responses using global indices
            response_idx = 0
            for i, sample in enumerate(batch_samples):
                global_idx = batch_start + i
                if i in valid_local_indices:
                    response = batch_responses[response_idx]
                    response_idx += 1
                    result, should_retry = self._process_batch_response_with_retry_flag(sample, response)
                    result['generation_time'] = batch_time / len(valid_prompts)
                    result['retry_count'] = 0
                    all_results_with_retry_flags[global_idx] = (result, should_retry)
                else:
                    result = {'error': 'Failed to build prompt', 'is_correct': False, 'retry_count': 0}
                    all_results_with_retry_flags[global_idx] = (result, False)
            
            pbar.update(len(batch_samples))
            
            # Print progress every 10 batches
            # Note: all_results_with_retry_flags contains (result, should_retry) tuples or None
            initial_correct = sum(1 for item in all_results_with_retry_flags if item is not None and item[0].get('is_correct', False))
            initial_done = sum(1 for item in all_results_with_retry_flags if item is not None)
            if (batch_end // self.batch_size) % 10 == 0 and initial_done > 0:
                current_accuracy = initial_correct / initial_done
                pbar.set_postfix({'accuracy': f'{current_accuracy:.2%}'})
        
        pbar.close()
        
        # Phase 2: Global batch retry for all failed extractions
        # Collect all samples that need retry (using global indices)
        for retry_round in range(1, self.max_retries + 1):
            retry_global_indices = [
                i for i, (result, should_retry) in enumerate(all_results_with_retry_flags)
                if should_retry and all_prompts[i] is not None
            ]
            
            if not retry_global_indices:
                break  # No more retries needed
            
            print(f"\nGlobal retry round {retry_round}/{self.max_retries}: {len(retry_global_indices)} samples need retry")
            
            # Prepare retry data
            retry_prompts = [all_prompts[i] for i in retry_global_indices]
            retry_samples = [samples[i] for i in retry_global_indices]
            
            # Process retries in batches for efficiency
            retry_temp = temperature * (1 + retry_round * 0.1)
            
            for retry_batch_start in range(0, len(retry_global_indices), self.batch_size):
                retry_batch_end = min(retry_batch_start + self.batch_size, len(retry_global_indices))
                retry_batch_prompts = retry_prompts[retry_batch_start:retry_batch_end]
                retry_batch_samples = retry_samples[retry_batch_start:retry_batch_end]
                retry_batch_global_indices = retry_global_indices[retry_batch_start:retry_batch_end]
                
                retry_start = time.time()
                try:
                    retry_responses = self.generate_responses_batch(retry_batch_prompts, retry_temp)
                except Exception as e:
                    if verbose:
                        print(f"  Retry batch generation failed: {e}")
                    continue
                retry_time = time.time() - retry_start
                total_time += retry_time
                
                # Update results for retried samples
                for j, global_idx in enumerate(retry_batch_global_indices):
                    if j < len(retry_responses):
                        old_result, _ = all_results_with_retry_flags[global_idx]
                        result, should_retry = self._process_batch_response_with_retry_flag(
                            retry_batch_samples[j], retry_responses[j]
                        )
                        result['generation_time'] = old_result.get('generation_time', 0) + retry_time / len(retry_batch_prompts)
                        result['retry_count'] = retry_round
                        all_results_with_retry_flags[global_idx] = (result, should_retry)
        
        # Phase 3: Finalize results
        all_results = []
        predictions = []
        
        for global_idx, (result, _) in enumerate(all_results_with_retry_flags):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Sample {global_idx}:")
                print(f"Selected: [{result.get('selected_label', '')}] {result.get('selected_title', '')}")
                print(f"Expected: [{result.get('expected_label', '')}] {result.get('expected_title', '')}")
                print(f"Correct: {result.get('is_correct', False)}")
                if result.get('retry_count', 0) > 0:
                    print(f"Retries: {result['retry_count']}")
            
            all_results.append(result)
            predictions.append(result.get('is_correct', False))
        
        # Calculate overall metrics
        accuracy = sum(predictions) / len(predictions) if predictions else 0.0
        avg_time = total_time / total_samples if total_samples > 0 else 0
        
        overall_metrics = {
            'total_samples': total_samples,
            'correct_predictions': sum(predictions),
            'accuracy': accuracy,
            'average_generation_time': avg_time,
            'total_time': total_time,
            'batch_size': self.batch_size
        }
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total samples: {overall_metrics['total_samples']}")
        print(f"Correct predictions: {overall_metrics['correct_predictions']}")
        print(f"Accuracy: {overall_metrics['accuracy']:.2%}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per sample: {avg_time:.2f}s")
        print(f"Throughput: {total_samples / total_time:.1f} samples/sec" if total_time > 0 else "N/A")
        
        # Save results if output path provided
        if output_path:
            results_data = {
                'overall_metrics': overall_metrics,
                'individual_results': all_results,
                'config': {
                    'model_path': self.model.__class__.__name__,
                    'temperature': temperature,
                    'max_length': self.max_length,
                    'max_new_tokens': self.max_new_tokens,
                    'max_retries': self.max_retries,
                    'similarity_threshold': self.similarity_threshold,
                    'if_shuffle_candidates': if_shuffle_candidates,
                    'random_seed': random_seed,
                    'data_file': data_file_path,
                    'overlapping_dir': overlapping_dir,
                    'batch_size': self.batch_size
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"\nResults saved to {output_path}")
        
        return overall_metrics


def main():
    """Main function to run evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate inspiration retrieval model')
    
    # Model configuration
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to base model')
    parser.add_argument('--lora_path', type=str, default=None,
                       help='Path to LoRA checkpoint (optional, for fine-tuned model)')
    
    # Data configuration
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to raw data JSON file (collected_inspiration_retrieval_QA_data_*.json). '
                            'This contains ALL samples for evaluation.')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save evaluation results')
    parser.add_argument('--no_shuffle', action='store_true',
                       help='Do not shuffle candidate order. Useful for reproducibility.')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Random seed for reproducible candidate shuffling. '
                            'If not set, shuffling is random. Ignored if --no_shuffle is used.')
    parser.add_argument('--overlapping_dir', type=str, default=None,
                       help='Path to directory containing overlapping train/eval samples to filter out. '
                            'Files in this directory should be named like "YYYY_PMID.json".')
    
    # Evaluation settings
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--start_idx', type=int, default=None,
                       help='Start index for parallel processing (0-based, inclusive)')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='End index for parallel processing (exclusive)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Generation temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top-p sampling parameter')
    parser.add_argument('--max_length', type=int, default=16384,
                       help='Maximum sequence length')
    parser.add_argument('--max_new_tokens', type=int, default=4096,
                       help='Maximum new tokens to generate')
    parser.add_argument('--max_retries', type=int, default=2,
                       help='Maximum retry attempts for poor extractions (default: 2, reduced for efficiency)')
    parser.add_argument('--similarity_threshold', type=float, default=0.3,
                       help='Minimum acceptable similarity for extraction (default: 0.3, matches SFT data minimum)')
    parser.add_argument('--load_in_8bit', action='store_true',
                       help='Load model in 8-bit precision to save memory')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed output for each sample')
    parser.add_argument('--debug_template', action='store_true',
                       help='Print template debug information')
    
    # Performance tuning
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for generation (higher = better GPU utilization). '
                            'Recommended: 32-48 for 7B models, 8-16 for 32B models')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = InspirationRetrievalEvaluator(
        model_path=args.model_path,
        lora_path=args.lora_path,
        device="cuda",
        load_in_8bit=args.load_in_8bit,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
        similarity_threshold=args.similarity_threshold,
        debug_template=args.debug_template,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # Run evaluation
    evaluator.evaluate_dataset(
        data_file_path=args.data_file,
        output_path=args.output_path,
        max_samples=args.max_samples,
        temperature=args.temperature,
        verbose=args.verbose,
        if_shuffle_candidates=not args.no_shuffle,
        random_seed=args.random_seed if not args.no_shuffle else None,
        overlapping_dir=args.overlapping_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )


if __name__ == "__main__":
    main()
