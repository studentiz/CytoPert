---
name: perturbation-de
description: Standard differential expression pipeline for perturbation vs control across cell states
version: 0.1.0
metadata:
  cytopert:
    category: pipelines
    tags: [scanpy, pertpy, de, perturbation]
    requires_tools: [scanpy_preprocess, scanpy_de]
---

# Perturbation DE

## When to Use
- You have an AnnData with both a perturbation/condition column (e.g. `perturbation`, `condition`) and a state/cluster column (e.g. `cell_type`, `state_groups`).
- Goal: produce a top-N DE gene list per state to feed downstream `decoupler_enrichment` and the `chains` tool.

## Procedure
1. Run `scanpy_preprocess path=<h5ad>` (defaults: `min_genes=200`, `min_cells=3`, `n_top_genes=2000`, `n_pcs=50`) — saves `scanpy_preprocessed.h5ad` to workspace.
2. (optional) `scanpy_cluster path=<workspace>/scanpy_preprocessed.h5ad method=leiden resolution=1.0` if state labels are missing.
3. For each state group, run `scanpy_de path=... groupby=<state_col> group1=<perturbation> group2=<control>`. Record top genes per state.
4. Feed the top genes into `decoupler_enrichment` for pathway hypotheses.
5. Call the `evidence` tool to confirm DE entries were added; cite their evidence IDs in the next mechanism chain.

## Pitfalls
- `wilcoxon` is the default — switch to `t-test_overestim_var` only when groups are very unbalanced.
- If `groupby` is not a categorical column, scanpy silently coerces; cast to `str` first to keep group labels stable.
- Don't average DE results across very different cell states — that's exactly what CytoPert is supposed to avoid.

## Verification
- Top genes contain at least one gene already implicated in the relevant pathway (verify with the upcoming `pathway_lookup` tool once it lands; do not assume the legacy `decoupler_enrichment` tool is still registered).
- Evidence IDs returned by `evidence` follow the live `tool_<tool_name>_<digest>` scheme (e.g. `tool_scanpy_de_3a4b5c6d7e`); always copy the id from the tool output instead of guessing a prefix like `de_*`.
