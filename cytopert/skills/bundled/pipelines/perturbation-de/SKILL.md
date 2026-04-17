---
name: perturbation-de
description: Standard differential expression pipeline for any condition vs control across cell states
version: 0.2.0
metadata:
  cytopert:
    category: pipelines
    tags: [scanpy, de, perturbation, pathway_lookup]
    # Only the scanpy steps are strictly required to surface the SKILL;
    # pathway_lookup is an *optional* downstream step (the SKILL still
    # produces useful DE evidence without it). Listing it in
    # ``requires_tools`` would hide the SKILL whenever pathway_lookup
    # is disabled by a check_fn or unavailable in a custom registry.
    requires_tools: [scanpy_preprocess, scanpy_de]
---

# Perturbation DE

## When to Use
- You have an AnnData with both a perturbation / condition column (e.g. `perturbation`, `condition`, `treatment`) and a state / cluster column (e.g. `cell_type`, `state_groups`).
- Goal: produce a top-N DE gene list per state and translate the gene list into pathway / TF hypotheses recorded as evidence.

## Procedure
1. Run `scanpy_preprocess path=<h5ad>` (defaults: `min_genes=200`, `min_cells=3`, `n_top_genes=2000`, `n_pcs=50`) — saves `scanpy_preprocessed.h5ad` to workspace.
2. (optional) `scanpy_cluster path=<workspace>/scanpy_preprocessed.h5ad method=leiden resolution=1.0` if state labels are missing.
3. For each state group, run `scanpy_de path=... groupby=<state_col> group1=<perturbation> group2=<control>`. Record top genes per state.
4. Feed the top genes into `pathway_lookup genes=[...] source=progeny` (or `dorothea` / `collectri`) to obtain pathway / TF hypotheses; the result is automatically recorded as KNOWLEDGE evidence.
5. Call the `evidence` tool to confirm DE + pathway entries were added; cite their evidence IDs in the next mechanism chain.

## Pitfalls
- `wilcoxon` is the default DE method — switch to `t-test_overestim_var` only when groups are very unbalanced.
- If `groupby` is not a categorical column, scanpy silently coerces; cast to `str` first to keep group labels stable.
- Don't average DE results across very different cell states — that's exactly what CytoPert is supposed to avoid.
- `pathway_lookup` returns lower-cased target matches (case-insensitive); supply the canonical symbol (e.g. mouse `Nfatc1`, human `NFATC1`) and let the lookup normalise.

## Verification
- Top genes overlap with at least one regulator returned by `pathway_lookup` (PROGENy / DoRothEA / CollecTRI), recorded as a KNOWLEDGE evidence entry.
- Evidence IDs returned by `evidence` follow the live `tool_<tool_name>_<digest>` scheme (e.g. `tool_scanpy_de_3a4b5c6d7e`, `tool_pathway_lookup_9b0a1c2d3e`); always copy the id from the tool output instead of guessing a prefix.
