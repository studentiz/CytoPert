---
name: census-tissue-slice
description: Generic procedure for slicing a cellxgene Census tissue with QC defaults
version: 0.2.0
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
1. Pin a Census version (set the most recent published release the
   researcher trusts) and use `obs_only=true` first to scout cell counts.
   Replace `<UBERON_ID>` and `<CENSUS_VERSION>` with the values for the
   actual tissue and Census release; the example below is shape-only and
   does not bind CytoPert to any specific tissue:
   ```text
   census_query obs_value_filter="tissue_ontology_term_id == '<UBERON_ID>'" obs_only=true max_cells=20000 census_version=<CENSUS_VERSION> timeout_seconds=60
   ```
   For example, `UBERON:0001911` is mammary gland, `UBERON:0002048`
   is lung, `UBERON:0000178` is whole blood. Look the right id up in
   the [UBERON browser](http://obofoundry.org/ontology/uberon.html)
   for the system the researcher actually wants.
2. If counts look reasonable (`n_obs` in expected range), repeat **without** `obs_only` to fetch the AnnData slice (`max_cells` 5k-20k for first pass).
3. Always pass `census_version` so the evidence is reproducible.
4. After loading, immediately call `evidence` (or `evidence_search`) to confirm the slice has been recorded as an EvidenceEntry.

## Pitfalls
- Forgetting `census_version` makes evidence non-reproducible — refuse to cite if missing.
- `tissue_ontology_term_id` is preferred over `tissue` string (avoids ambiguous matches across species and curated synonyms).
- Census timeouts often signal too-broad filters; tighten by adding `disease`, `assay`, or `cell_type_ontology_term_id`.
- Different Census releases ship different `tissue_ontology_term_id` coverage; if a known-good id returns 0 cells, double-check against `dc.op.show_organisms()` and the Census release notes.

## Verification
- `n_obs` matches the pre-flight `obs_only` probe within ~10%.
- `evidence_search query="<tissue or disease term>"` returns the new entry within seconds.
