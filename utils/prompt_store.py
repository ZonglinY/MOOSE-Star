# =============================================================================
# Prompt Store for MOOSE-Star Pipeline
# =============================================================================
# DeepSeek R1-Distill models are trained WITHOUT system prompts.
# All instructions are in the user prompt.
# The <think> tag is added by the chat template (add_generation_prompt=True).


def instruction_prompts(module_name):
    if module_name == "prepare_HC_sft_data_to_go_comprehensive_v2_delta":
        # Delta hypothesis output: THIS inspiration's contribution only
        # Structured format: Inspiration / Motivation / Mechanism / Methodology
        prompts = [
            "You are a scientific hypothesis composer. Given a research question, background survey, potentially a previous hypothesis to build upon, and a new inspiration paper, you will reason through how to adapt concepts from the inspiration to advance the solution, then formulate a DELTA HYPOTHESIS - the specific contribution from THIS inspiration paper (not the full cumulative hypothesis).\n\n## Your Task\n\nAnalyze the provided research context and inspiration paper to:\n1. Identify the key conceptual innovation from the inspiration paper (Note: the paper may directly provide a concept that can be adapted, OR it may contain related ideas/transferable mechanisms that inspire what we need - look beyond exact concept names)\n2. Determine how this innovation addresses gaps (either in existing methods or in your previous hypothesis)\n3. Reason through adaptation and integration into your solution\n4. Formulate a delta hypothesis describing ONLY what THIS inspiration contributes\n\n## Key Principles\n\n**Reasoning Process:**\n- Start by understanding the problem and what current methods lack\n- If there's a previous hypothesis, understand what it already addresses and what gaps/limitations remain\n- Analyze the inspiration paper to identify relevant concepts that could be adapted\n- Reason through: What specific knowledge/technique from this paper could serve as an inspiration?\n- Connect the dots: How does this potential inspiration address the identified gaps?\n- Work through the mechanism: How would this inspiration actually function in our context?\n- Develop the methodology: Detail the specific implementation and integration\n- Don't just identify concepts - reason through their practical application and adaptation\n\n**Delta Hypothesis Requirements:**\n- Output ONLY what THIS inspiration adds (delta), NOT the full cumulative hypothesis\n- Don't repeat what's already in the previous hypothesis\n- Must clearly explain WHY the inspiration addresses the problem (Motivation)\n- Must detail HOW the inspiration works in this context (Mechanism)\n- Must specify HOW to implement it methodologically (Methodology)\n- Follow the exact structured format shown below\n\n## Information Provided\n\n**Research Question** (the specific problem to solve):\n",
            "\n\n**Background Survey** (existing methods and their limitations):\n",
            "\n\n**Previous Hypothesis** (if any - the current state of your solution to build upon):\n",
            "\n\n**New Inspiration Paper Title** (external work to incorporate):\n",
            "\n\n**New Inspiration Paper Abstract**:\n",
            "\n\n## Your Response\n\nAnalyze how this inspiration paper's concepts can advance your solution:\n\n1. **If starting from scratch** (no previous hypothesis): \n   - Identify how the inspiration addresses the core gaps in existing methods\n   - This becomes your first conceptual building block beyond the baseline approach\n\n2. **If building on a previous hypothesis**: \n   - First understand what the previous hypothesis already accomplishes\n   - Identify remaining limitations or opportunities for enhancement\n   - Determine how this new inspiration specifically addresses those gaps\n\nShow your reasoning process as you:\n- Extract relevant concepts from the inspiration paper (may not be obvious - reason through what could be useful)\n- Identify what specific technique/knowledge could serve as the inspiration\n- Connect this inspiration to your problem: Why is this relevant? How does it address gaps?\n- Work through adaptation: How to modify this concept for your specific context?\n- Detail the motivation, mechanism, and methodology for THIS inspiration's contribution\n- Reason through implementation details\n\nThen formulate a delta hypothesis that captures ONLY what THIS inspiration adds.\n\n## Output Format\n\n**IMPORTANT**: Structure your response exactly as follows:\n\n<think>\n[Your reasoning process here - explore all aspects thoroughly]\n</think>\n\n**Delta Hypothesis starts:**\nInspiration: [Key concept derived from or inspired by the inspiration paper]\n- Motivation (WHY): [Why this addresses a gap - what specific limitation does it solve?]\n- Mechanism (HOW IT WORKS): [How the concept works in this context]\n- Methodology (HOW IT'S INTEGRATED): [How to integrate it - specific implementation steps]\n**Delta Hypothesis ends**\n\n⚠️ CRITICAL: The delta hypothesis is the ONLY part that gets evaluated!\n- Include ALL components you FINALIZED in your reasoning (not early ideas you later revised)\n- Be COMPREHENSIVE - every technical detail, mechanism, and methodology step you reasoned through should appear in the delta hypothesis\n- Don't assume the reader saw your reasoning - the delta hypothesis must be SELF-CONTAINED and COMPLETE\n- Focus on THIS inspiration's contribution only - don't repeat previous hypothesis content"
        ]
    elif module_name == "inspiration_retrieval_with_reasoning_with_alphabetical_candidates":
        prompts = [
            "You are helping with scientific hypothesis generation by selecting an inspiration that solves a fundamental problem in the current approach.\n\n## Core Task: Problem Identification and Solution\n\n**Your Primary Goal**: Identify which candidate paper can best help solve a fundamental problem in the existing methods/hypothesis - either directly or by inspiring a solution.\n\n**Key Principle**: Good inspirations help solve real problems. They might directly provide a solution, or they might spark an idea, remind you of related concepts, or inspire a creative adaptation. The best breakthroughs often come from unexpected connections.\n\n**What Makes a Good Inspiration**:\n1. **Problem-Solution Fit**: Either addresses a known limitation OR reveals new improvement opportunities\n2. **Enables Progress**: The paper provides concepts, sparks ideas, or inspires solutions that advance the research\n3. **Creative Connection**: The link might be indirect, non-obvious, or emerge during exploration\n4. **Clear Impact**: You can explain how this paper contributes to progress, even if the path is unexpected\n\n**The Research Process**:\n1. **Background**: Research question + existing methods (with their limitations)\n2. **Problem Identification**: What fundamental issue prevents progress?\n3. **Inspiration Selection**: Which concept best solves this problem?\n4. **Hypothesis Formation**: Adapt the solution to create a better method\n\n**Classic Example - Backpropagation**:\n- **Research Question**: How to use data to automatically improve parameters of multi-layer logistic regression?\n- **Existing Methods**: Could only do inference, not learning\n- **FUNDAMENTAL PROBLEM**: No way to compute gradients through multiple layers\n- **Solution Found**: Chain rule from calculus\n- **Why It Solves the Problem**: Chain rule computes derivatives of composite functions; neural networks ARE composite functions\n- **Result**: Backpropagation algorithm\n\nNote: The focus was on SOLVING THE GRADIENT PROBLEM. The breakthrough came from recognizing neural networks as composite functions.\n\n## Your Current Task\n\n**Flexible Reasoning Process** (these steps can happen in any order or iteratively):\n- **Problem Recognition**: Identify limitations in current methods/hypothesis (can happen before OR after seeing candidates)\n- **Opportunity Discovery**: For each candidate, explore how it might advance the research:\n   - It might solve a problem you already identified\n   - It might reveal a problem you hadn't noticed and simultaneously offer a solution\n   - It might spark ideas for improvements you hadn't considered\n- **Selection**: Choose the candidate that enables the most meaningful progress\n\n**Note**: The reasoning is often bidirectional - seeing a candidate can make you realize \"oh, this could address limitation X that I hadn't fully articulated\" or \"this suggests a way to improve aspect Y\"\n\n**Remember**:\n- The best inspiration might not seem immediately relevant\n- Focus on problem-solving potential, not keyword matching\n- Creative connections often lead to breakthroughs\n- Consider how concepts could be adapted or repurposed\n\n**Avoid**:\n- Choosing based on surface-level similarity\n- Dismissing candidates that seem unrelated at first glance\n\n## Context\n\n**Research Question:**\n",
            "\n\n**Background Survey (existing methods for THIS task):**\n",
            "\n\n**Previous Hypothesis (if any - current progress built from earlier inspirations):**\n",
            "\n\n## Candidate Inspiration Papers\n\n",
            "\n\n## Output Format\n\n**CRITICAL**: You MUST structure your response EXACTLY as follows (the markers are used for automatic parsing).\n\n<think>\n[Your flexible reasoning process - explore problems and opportunities as they emerge, evaluate how candidates relate to potential improvements, select the most promising one. Refer to candidates using their labels like Candidate [A], Candidate [B], etc.]\n</think>\n\n**Selected ID starts:** [X] **Selected ID ends**\n\n(Replace [X] with the letter of your chosen candidate, e.g., [A], [B], [C], etc. Output ONLY the letter in brackets, nothing else between the markers.)\n\n**Selection Reason starts:** [summary of why this inspiration was selected - what problem it addresses, how it enables progress] **Selection Reason ends**"
        ]
    elif module_name == "generate_reasoning_trace_per_step_updated_recall":
        # Reasoning trace generation with balanced coverage emphasis for better recall
        prompts = [
            """You are tasked with GENERATING a simulated reasoning trace that shows how a researcher would naturally arrive at a hypothesis by combining background knowledge with an inspiration paper.

**CRITICAL**: This is a generation task - you need to CREATE a realistic reasoning trace as output. Write the trace as if the researcher doesn't know the answer yet and is genuinely working through the problem. You are not actually reasoning - you are generating text that simulates realistic reasoning.

**Context - Information Availability**:

What the researcher KNOWS (can use explicitly):
- The research question (the problem to solve)
- The survey (current methods and their limitations)
- The inspiration paper's title and abstract (discovered external work)
- Previous hypothesis if any (prior thinking on the problem that we can build upon)

What the researcher DOESN'T KNOW (must discover through reasoning):
- The final hypothesis (must arrive at this naturally)
- The specific inspiration concept to extract from the paper
- How exactly to adapt and apply it to the problem
- The integration methodology

Hidden guidance (for quality control only - NEVER mention these):
- Groundtruth inspiration: ensures you explore the right concept
- Groundtruth relation: ensures you understand the connection correctly
- Target hypothesis: ensures your reasoning leads to the right solution

Your Generation Task: CREATE a simulated reasoning trace that recreates how a researcher would discover and adapt external concepts to solve the research problem. The output should read as if someone is finding the solution for the first time.

**Generation Requirements**:

1. **Generate Authentic Discovery Process**: Create text that appears as if discovering the solution in real-time
   - Start uncertain, explore possibilities
   - Show moments of realization ("Ah, this could work because...")
   - Include natural reflections and refinements
   - Don't jump straight to the perfect answer

2. **Natural Reasoning Flow**:
   - Begin by understanding the problem and gaps in current approaches
   - Explore what the inspiration paper offers
   - Reason through HOW to adapt it (not just that it could work)
   - Consider challenges and how to address them
   - Gradually build toward the hypothesis
   - **Work through the hypothesis details**: Don't just arrive at the idea - reason through the specifics of implementation

3. **Use Hidden Guidance Wisely**:
   - The groundtruth inspiration and relation are provided to ensure quality
   - Use them as guardrails, NOT as explicit knowledge
   - Let them guide your reasoning toward the right direction naturally
   - Don't mention you know these - discover them through reasoning

4. **Reasoning Depth WITH Comprehensive Coverage**:
   - WHY this inspiration addresses the problem (motivation) - reason through the specific gaps it fills
   - HOW it works conceptually (mechanism) - explore the underlying principles and why they apply
   - HOW to integrate it practically (methodology) - think through concrete implementation steps
   - What specific methodological details are needed - work out the practical aspects
   - What challenges might arise and how to handle them

   **IMPORTANT - Coverage Requirements**:
   - Be THOROUGH: Identify all key technical components, not just the main innovation
   - Include supporting mechanisms, data flow, and architectural elements
   - Think: "What would someone need to actually implement this?"
   - Balance completeness with conciseness - avoid redundancy

5. **Realistic Imperfection**:
   - Initial ideas might be partially formed or wrong - that's natural!
   - Show trial-and-error: "Actually, on second thought..." is realistic
   - Early ideas can be revised or discarded as understanding deepens
   - What matters is that your FINAL understanding is complete

6. **Balance Depth and Breadth**:
   - Write like a thorough technical review: deep insights with broad coverage
   - Explain WHY (motivation), HOW (mechanism), and WHAT (implementation)
   - Include all important components but keep explanations focused
   - Quality over quantity - better to explain fewer components well than list many superficially

**Enhanced Example Reasoning Pattern**:
"Looking at this problem of [X], current methods struggle with [Y]. This paper about [inspiration domain] is interesting - they solve a related challenge using [concept]. Hmm, could we adapt this? The key insight seems to be [mechanism].

If we apply this to our problem... wait, we'd need to modify it because [difference]. Actually, what if we [adaptation]? That could address [specific gap] because [reasoning].

Let me work through the key components:
- Core mechanism: [main innovation] works by [explanation]
- Supporting components: [component 2] for [reason], [component 3] for [reason]
- Integration: These connect via [how they work together]
- Implementation details: [specific technical aspects]
- Edge case handling: [how the method ensures robustness]

This would mean [concrete specification with all parts covered]..."

**Important Notes**:
- Generate reasoning that someone could realistically have WITHOUT knowing the answer
- Include the messy middle - not every thought needs to be correct immediately
- Show the 'aha' moments naturally emerging from exploration
- The reasoning can explore and revise ideas - early thoughts may be superseded by better ones

**CRITICAL for Final Hypothesis**:
- The hypothesis is what gets evaluated - it must be COMPLETE and SELF-CONTAINED
- Include ALL finalized components from your reasoning (not early ideas you later revised)
- If you reasoned "we need X, Y, and Z", then X, Y, and Z MUST appear in the hypothesis
- Don't assume readers saw your reasoning - the hypothesis should stand alone
- Think of it as: "If someone only read the hypothesis, would they know all the key components?"

Research Question:
""",
            "\n\nSurvey Context (what we already know about this problem):\n",
            "\n\nPrevious Hypothesis (if any, showing our current thinking that we can further build upon based on the information provided with the inspiration paper):\n",
            "\n\nInspiration Paper Title:\n",
            "\n\nInspiration Paper Abstract:\n",
            "\n\n[Hidden: Groundtruth Inspiration - use as implicit guidance only]:\n",
            "\n\n[Hidden: Groundtruth Relation Between the Inspiration and the Paper - use as implicit guidance only]:\n",
            "\n\n[Hidden: Target Hypothesis - what we should naturally arrive at]:\n",
            """

Now GENERATE a simulated reasoning trace that shows how a researcher would naturally arrive at the hypothesis by exploring how the inspiration paper could address the research question.

Remember: This is a text generation task. Create output that appears as if someone is discovering this solution for the first time, even though you have the hidden guidance to ensure quality.

**CRITICAL OUTPUT REQUIREMENTS**:

Your output should:
1. Show thorough reasoning that explores the problem deeply
2. Be COMPREHENSIVE in identifying all technical components needed
3. Balance conceptual understanding with technical completeness
4. Include all relevant mechanisms, not just the primary innovation
5. Consider supporting components and infrastructure needed
6. Address how different parts work together

**CRITICAL: You MUST strictly follow the output template below (so that we can extract your answer with rule-based extraction). Your entire response should consist of ONLY these two sections with the exact markers shown:**

**Simulated Reasoning Trace starts:**
[Generate natural, exploratory reasoning text here - should be substantial, showing a genuine thought process with uncertainty, exploration, and gradual understanding. IMPORTANTLY: Be comprehensive in working through ALL technical components, mechanisms, and methodological details. Don't just focus on the main idea - include supporting elements, data flow, processing steps, and how everything integrates. Think like you're preparing someone to actually implement this.]
**Simulated Reasoning Trace ends**

**Hypothesis starts:**
[Generate the hypothesis that naturally emerges from the reasoning trace - should capture BOTH the core insights about WHY this approach makes sense, HOW it works mechanistically, and HOW to implement it methodologically.

⚠️ CRITICAL: This hypothesis is the ONLY part that gets evaluated!
- Include ALL components you FINALIZED in your reasoning (not early ideas you later revised)
- If you concluded "we need X because Y", then X MUST appear here with its purpose
- If you explored idea A but then decided B is better, only include B
- The hypothesis must be SELF-CONTAINED - readers won't see your reasoning trace

Be COMPREHENSIVE - include all technical components, their interactions, supporting mechanisms, and implementation details. The hypothesis should be complete enough that someone could understand all the moving parts, not just the main innovation.]
**Hypothesis ends**

**DO NOT include any text outside of these two sections. DO NOT add any preamble, explanation, or commentary before or after these sections.**"""
        ]
    elif module_name == "generate_reasoning_trace_per_step_v2_delta":
        # Delta hypothesis output with exact label matching
        prompts = [
            """You are tasked with GENERATING a simulated reasoning trace that shows how a researcher would naturally arrive at a DELTA HYPOTHESIS - the specific contribution from ONE inspiration paper to the overall hypothesis.

**CRITICAL CONTEXT - Delta vs Cumulative Hypothesis**:
- You are generating the hypothesis contribution from a SINGLE inspiration (delta)
- NOT the full cumulative hypothesis that includes all previous inspirations
- The previous hypothesis (if any) provides context of what we've built so far
- Your output is ONLY what THIS inspiration adds to the hypothesis

**Context - Information Availability**:

What the researcher KNOWS (can use explicitly):
- The research question (the problem to solve)
- The survey (current methods and their limitations)
- Previous hypothesis if any (cumulative context from previous inspirations)
- The inspiration paper's title and abstract (discovered external work)

What the researcher DOESN'T KNOW (must discover through reasoning):
- The specific delta hypothesis (what THIS inspiration contributes)
- The specific concept/insight to derive from reading the inspiration paper
- How exactly to adapt and apply it to the research problem
- The relation between the inspiration and the research problem

Hidden guidance (for quality control only - these guide your reasoning direction, but NEVER mention them explicitly):
- Groundtruth inspiration: The key concept derived from the paper - use this only to ensure you explore the RIGHT concept, but discover it naturally through reading the abstract
- Groundtruth relation: How it connects to the problem - this is UNKNOWN at inference, so your reasoning must discover this connection naturally
- Target delta hypothesis: The final answer - this is UNKNOWN at inference, so reason toward it naturally without revealing you know it

Your Generation Task: CREATE a simulated reasoning trace showing how a researcher discovers what THIS SPECIFIC INSPIRATION contributes to solving the research problem.

**Generation Requirements**:

1. **Generate Authentic Discovery Process**:
   - Start by reviewing what we already have: the research question, the survey context, and any previous hypothesis
   - Explore what NEW the inspiration paper offers by reading its title and abstract
   - Reason through how concepts from the inspiration paper could help further improve the previous hypothesis (if any) to better resolve the research question
   - Show the specific contribution emerging naturally through exploration
   - **CRITICAL - Cover ALL Details**: Your reasoning trace MUST discuss and develop ALL details that will appear in the delta hypothesis. Every specific point in the final output (motivation, mechanism, methodology) should be discovered and reasoned through in the trace FIRST. Nothing should appear "out of nowhere" in the delta hypothesis.
   - **Avoid premature convergence**: Don't rush to the solution - spend time genuinely exploring the inspiration paper. The "aha moment" should come after sufficient exploration, not immediately.
   - **Self-correction pattern**: First reason from ONLY the observable information (research question, survey, previous hypothesis, inspiration abstract). This initial reasoning may lead to tentative or imperfect conclusions. Then, as you explore deeper, show moments of self-correction: "Initially I thought X might work, but looking more carefully at [specific evidence], I realize Y is more appropriate because..." This makes the trace look like natural discovery with refinement, not perfect reasoning from the start.
   - **Include the messy middle**: Initial ideas might be partially formed or wrong - that's natural! Show trial-and-error thinking. Early ideas can be revised or discarded as understanding deepens. The "aha moment" should emerge naturally from exploration.

2. **Natural Reasoning Flow**:
   - Understand current gaps given the previous hypothesis
   - Explore the inspiration paper's key concepts
   - Reason through HOW this inspiration addresses remaining gaps
   - Work through the specific contribution this inspiration makes
   - **Ground observations in specific text**: When exploring the inspiration paper, quote or reference specific phrases from the abstract. Write "The abstract mentions [specific phrase]" rather than "The paper proposes [summary]".

3. **Reasoning Depth for Delta Hypothesis** (derive EVERY detail through reasoning):
   - WHY this inspiration addresses a specific gap (Motivation) - reason through the logical connection
   - HOW the concept works (Mechanism) - explain the underlying principle step by step
   - HOW to integrate it practically (Methodology) - **derive specific approaches by reasoning**: "Based on [abstract's mention of X], a reasonable approach would be [specific method] because [reasoning]. To implement this, we would need [step 1], [step 2]..." Don't just state methodology - REASON your way to each step.

4. **Focus on THIS Inspiration's Contribution**:
   - Don't repeat what's already in the previous hypothesis
   - Focus on what's NEW from this inspiration
   - The delta should be a self-contained contribution

5. **Language Patterns for Authentic Discovery**:
   - Prefer exploratory language: "I notice that...", "This makes me wonder...", "Could this relate to...", "Looking at the abstract, I see..."
   - Avoid answer-revealing language: "The key insight is...", "This clearly shows...", "The answer is...", "Obviously..."
   - The trace should read like thinking-out-loud, not explaining a known answer

6. **Coverage and Completeness**:
   - Be thorough: identify all technical components for THIS delta, not just the main idea

**Example Reasoning Pattern** (for reference - showing self-correction, exploration, and detailed reasoning):
"Let me understand where we are. [Review research question and survey. If previous hypothesis exists, what does it address? What remains unsolved?]

Now looking at this inspiration paper... The title '[title]' is about [domain]. I notice the abstract mentions '[quote phrase]'. This makes me wonder - could there be a connection? They're working on [different problem], but let me explore further.

The abstract says '[quote another phrase]'. Initially I thought this was just about [surface interpretation], but looking at '[quote specific detail]', I'm starting to see something deeper...

Hmm, wait - could this actually help with our problem? Let me think about what gap we still have. [Identify limitation]. The abstract mentions '[relevant quote]'... at first I wasn't sure how this relates, but now I see the connection: [reasoning].

Actually, I need to reconsider my first instinct. I initially thought [initial idea], but that doesn't quite fit because [reason]. Looking back at '[quote]', a better interpretation is [refined understanding]. This makes more sense because [reasoning].

Let me work through the details carefully:

WHY would this help (motivation)? [Don't just state - reason through it. What specific gap does this address? Why is the abstract's approach relevant to that gap? Connect '[quote]' to the limitation we identified.]

HOW does it work (mechanism)? The abstract hints at this with '[quote technical aspect]'. The core principle seems to be [mechanism reasoning]. I think this works because [explain the underlying logic - why this principle is effective for the problem].

HOW to integrate it (methodology)? The abstract mentions '[quote about approach]'. Let me think through how to implement this step by step:
- First, we would need [step 1] because [reasoning why this step is necessary based on the mechanism]
- Then, [step 2] would follow because [reasoning connecting to previous step]
- For measurement/validation, [step 3] makes sense because [reasoning based on what abstract suggests]
- The abstract's mention of '[quote]' suggests we should [specific approach] rather than [alternative]

Putting this together: this inspiration offers [concept] which addresses [specific gap] because [motivation summary]. It works by [mechanism summary] and we integrate it through [methodology summary - each step grounded in the reasoning above]..."

**CRITICAL - Output Format**:
The delta hypothesis MUST follow this exact structure:
```
Inspiration: [Key concept derived from or inspired by the inspiration paper]
- Motivation (WHY): [Why this addresses a gap]
- Mechanism (HOW IT WORKS): [How the concept works - NO novel method names]
- Methodology (HOW IT'S INTEGRATED): [How to integrate it - NO novel method names]
```

**CRITICAL - No NOVEL Method Names in Mechanism/Methodology**:
- The "Inspiration:" line describes the key concept derived from the inspiration paper - concept names here are FINE
- Classic/well-known method names are FINE everywhere: BERT, CNN, LDA, Transformer, etc.
- What to AVOID: Novel method names in the Mechanism or Methodology sections that refer to the NEW hypothesis being proposed
- Reason: The model cannot predict novel method names at inference time
- BAD in Mechanism: "The MolFormer architecture processes molecules by..."
- GOOD in Mechanism: "The molecular graph neural network processes molecules by..."
- BAD in Methodology: "We implement this as our HyperDrug (HD) framework..."
- GOOD in Methodology: "We implement this using hypergraph-based drug interaction modeling..."

Research Question:
""",
            "\n\nSurvey Context (what we already know about this problem):\n",
            "\n\nPrevious Hypothesis (cumulative context from previous inspirations - build upon this):\n",
            "\n\nInspiration Paper Title:\n",
            "\n\nInspiration Paper Abstract:\n",
            "\n\n[Hidden: Groundtruth Inspiration - use as implicit guidance only]:\n",
            "\n\n[Hidden: Groundtruth Relation - use as implicit guidance only]:\n",
            "\n\n[Hidden: Target Delta Hypothesis - the EXACT contribution from this inspiration that you should arrive at]:\n",
            """

Now GENERATE a simulated reasoning trace that shows how a researcher discovers what THIS SPECIFIC INSPIRATION contributes.

**REASONING TRACE REQUIREMENTS** (what to include in your reasoning):
- What gap remains given the previous hypothesis?
- What does this inspiration paper offer? Quote specific phrases from the abstract. (Note: the abstract may not always contain the exact concept name - look for related ideas or transferable mechanisms)
- How does it address the remaining gap?
- Work through the motivation, mechanism, and methodology for THIS contribution.
- **For methodology: DERIVE each step through reasoning** - don't just state steps, explain WHY each step follows from the mechanism and abstract. Example: "Since the abstract mentions [X], we would need to [step] because [reasoning]..."
- Use exploratory language: "I notice...", "This makes me wonder...", "Looking at the abstract, I see..."
- **Cover ALL details** - every specific point in your delta hypothesis MUST be derived/reasoned in the trace first
- Don't rush to the answer - explore first, then converge
- NEVER mention the hidden guidance (groundtruth inspiration, relation, or target hypothesis) - discover everything naturally from the observable inputs
- Ensure the reasoning trace naturally and coherently leads to the delta hypothesis - the conclusion should feel earned through the exploration

**DELTA HYPOTHESIS REQUIREMENTS**:
- Focus on what THIS inspiration adds (delta), not the full cumulative hypothesis
- Follow the exact structure: Inspiration / Motivation / Mechanism / Methodology
- Match the target delta hypothesis in content and structure
- Replace any novel method names (for the proposed hypothesis) in Mechanism/Methodology with generic descriptions
- Concept names in the "Inspiration:" line are fine; classic names like BERT, CNN are fine everywhere

**CRITICAL: Your output MUST contain these EXACT markers (copy them exactly):**

**Simulated Reasoning Trace starts:**
(your multi-paragraph reasoning trace)
**Simulated Reasoning Trace ends**

**Delta Hypothesis starts:**
Inspiration: ...
- Motivation (WHY): ...
- Mechanism (HOW IT WORKS): ...
- Methodology (HOW IT'S INTEGRATED): ...
**Delta Hypothesis ends**"""
        ]
    else:
        raise NotImplementedError("Module name not found: {}".format(module_name))

    return prompts
