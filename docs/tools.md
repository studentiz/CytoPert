# Tools

CytoPert tools are callable by the agent. Each tool returns a string summary and can be chained for analysis.

## Census Tools

### `census_query`

Query cellxgene Census and return a summary of the slice.

Common parameters:
- `obs_value_filter`: SOMA filter for cells (e.g. `tissue_ontology_term_id == 'UBERON:0000178'`)
- `var_value_filter`: SOMA filter for genes
- `census_version`: recommended for reproducibility (e.g. `2025-11-08`)
- `organism`: default `Homo sapiens`
- `obs_only`: `true` to fetch only metadata (faster)
- `max_cells`: default `20000`
- `timeout_seconds`: default `30`

Example (metadata only):

```
obs_value_filter="tissue_ontology_term_id == 'UBERON:0000178'" obs_only=true max_cells=20000 timeout_seconds=60
```

### `load_local_h5ad`

Load local `.h5ad` and return `n_obs`, `n_vars`, and obs columns.

## Scanpy Tools

- `scanpy_preprocess`
- `scanpy_cluster`
- `scanpy_de`

## Pertpy Tools

- `pertpy_perturbation_distance`
- `pertpy_differential_response`

## Decoupler Tools

- `decoupler_enrichment`

## Evidence

- `evidence` – summarise evidence store

## Pathway & Mechanism

- `pathway_constraint`
- `pathway_check`
- `chains`

