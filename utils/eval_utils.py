"""
Evaluation utilities for MOOSE-Star pipeline.
"""


def sample_one_MDP_for_one_paper_from_hypothesis_components(inspirations, hypothesis_components, paper_name):
    """
    Build MDP road from hypothesis_components.

    Always follows sequential order (0->1->2->...) since components are indexed.

    hypothesis_components: {"0": delta_0, "1": delta_1, ...}
    Returns: [[0, delta_0], [1, delta_1], ...] in sequential order
    """
    n = len(inspirations)
    assert n >= 1, f"There should be at least one inspiration: {paper_name}"
    assert n == len(hypothesis_components), \
        f"Mismatch: {n} inspirations vs {len(hypothesis_components)} hypothesis_components: {paper_name}"

    # Always use sequential order: 0 -> 1 -> 2 -> ...
    MDP_road = []
    for i in range(n):
        delta_hyp = hypothesis_components[str(i)]
        MDP_road.append([i, delta_hyp])

    return MDP_road
