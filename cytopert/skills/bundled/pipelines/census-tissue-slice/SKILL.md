---
name: census-tissue-slice
description: Query cellxgene Census for a tissue with QC defaults and cite as evidence
version: 0.1.0
metadata:
  cytopert:
    category: pipelines
    tags: [census, anndata, qc]
    requires_tools: [census_query]
---

# Census Tissue Slice

## When to Use
- The researcher names a tissue / disease / cell type and you have no local h5ad yet.
- Either a fresh exploratory question OR confirming previously cited evidence is still reachable.

## Procedure
1. Pin a Census version (default `2025-11-08`) and use `obs_only=true` first to scout cell counts:
   ```text
   census_query obs_value_filter="tissue_ontology_term_id == 'UBERON:0001911'" obs_only=true max_cells=20000 census_version=2025-11-08 timeout_seconds=60
   ```
2. If counts look reasonable (`n_obs` in expected range), repeat **without** `obs_only` to fetch the AnnData slice (`max_cells` 5k-20k for first pass).
3. Always pass `census_version` so the evidence is reproducible.
4. After loading, immediately call `evidence` (or `evidence_search`) to confirm the slice has been recorded as an EvidenceEntry.

## Pitfalls
- Forgetting `census_version` makes evidence non-reproducible — refuse to cite if missing.
- `tissue_ontology_term_id` is preferred over `tissue` string (avoids ambiguous matches).
- Census timeouts often signal too-broad filters; tighten by adding `disease`, `assay`, or `cell_type_ontology_term_id`.

## Verification
- `n_obs` matches the pre-flight `obs_only` probe within ~10%.
- `evidence_search query="<tissue>"` returns the new entry within seconds.
