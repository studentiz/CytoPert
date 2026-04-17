# Tools

Every tool below is registered by `AgentLoop._register_default_tools` in
`cytopert/agent/loop.py`. Tools removed in stage 1 of the completeness
overhaul (`pertpy_perturbation_distance`, `pertpy_differential_response`,
`decoupler_enrichment`, `pathway_check`, `pathway_constraint`) are no
longer reachable; the `pathway_lookup` tool that replaces the pathway
surface lands in stage 7.2.

## Census

### `census_query`

Query cellxgene Census and return a slice summary.

| Parameter            | Type     | Default          | Description                                                                                                       |
| -------------------- | -------- | ---------------- | ----------------------------------------------------------------------------------------------------------------- |
| `obs_value_filter`   | str      | -                | SOMA value filter for cells (e.g. `tissue_ontology_term_id == 'UBERON:0001911'`).                                 |
| `var_value_filter`   | str      | -                | SOMA value filter for genes.                                                                                      |
| `census_version`     | str      | latest           | Pin a published Census version for reproducibility (e.g. `2025-11-08`).                                           |
| `organism`           | str      | `Homo sapiens`   | Census organism.                                                                                                  |
| `obs_only`           | bool     | `false`          | Return only the obs metadata; faster for scouting cell counts.                                                    |
| `max_cells`          | int      | `20000`          | Cap the AnnData slice when not in `obs_only` mode.                                                                |
| `obs_coords`         | str      | -                | Optional explicit row coords forwarded to SOMA (`get_anndata`).                                                   |
| `timeout_seconds`    | int      | `30`             | Hard timeout for the SOMA query.                                                                                  |

`AgentLoop._maybe_parse_forced_tool_call` recognises text mentions of
`census_query` plus the keys above and forces the tool call directly
(useful when the model is reluctant to make a structured call).

### `load_local_h5ad`

Load a local `.h5ad`. Returns `n_obs`, `n_vars`, and the obs columns.

| Parameter | Type | Description                          |
| --------- | ---- | ------------------------------------ |
| `path`    | str  | Absolute path to a readable `.h5ad`. |

## Scanpy

### `scanpy_preprocess`

Standard scanpy preprocessing (`pp.filter_cells / filter_genes / normalize_total /
log1p / highly_variable_genes / scale / pp.pca`). Writes
`scanpy_preprocessed.h5ad` into the workspace.

| Parameter      | Type | Default | Description                                          |
| -------------- | ---- | ------- | ---------------------------------------------------- |
| `path`         | str  | -       | Path to the input `.h5ad`.                            |
| `min_genes`    | int  | `200`   | Minimum genes per cell.                              |
| `min_cells`    | int  | `3`     | Minimum cells per gene.                              |
| `n_top_genes`  | int  | `2000`  | HVG count.                                           |
| `n_pcs`        | int  | `50`    | Number of principal components.                      |

### `scanpy_cluster`

`pp.neighbors` then Leiden / Louvain clustering on the workspace AnnData.

| Parameter    | Type | Default   | Description                              |
| ------------ | ---- | --------- | ---------------------------------------- |
| `path`       | str  | -         | Path to the preprocessed `.h5ad`.        |
| `method`     | str  | `leiden`  | `leiden` or `louvain`.                   |
| `resolution` | float| `1.0`     | Resolution parameter.                    |

### `scanpy_de`

`tl.rank_genes_groups`. Reads the AnnData at `path` and returns a
ranked-genes summary string suitable for evidence extraction.

| Parameter | Type | Default | Description                                              |
| --------- | ---- | ------- | -------------------------------------------------------- |
| `path`    | str  | -       | Path to the preprocessed `.h5ad`. Required.              |
| `groupby` | str  | -       | obs column to group by (e.g. `condition`, `cell_type`). Required. |
| `group1`  | str  | -       | Group of interest. Required.                             |
| `group2`  | str  | -       | Reference group (must be a real group label in `groupby`; the tool does not currently expose a one-vs-rest shortcut). Required. |
| `top_n`   | int  | `20`    | How many top genes to include in the result string.      |

The DE method is hard-coded to `wilcoxon` in the current implementation.
Add an explicit `method` parameter to the tool's JSON schema if your
workflow needs `t-test_overestim_var` / `logreg`; the underlying scanpy
call is straightforward to extend.

## Pathway / TF lookup

### `pathway_lookup`

Look up the PROGENy / DoRothEA / CollecTRI regulators of a gene set
through `decoupler.op`. The first call per `(source, organism)` fetches
the network DataFrame and caches it under
`~/.cytopert/cache/knowledge/<source>__<organism>.parquet` so subsequent
calls are offline. The result is recorded as KNOWLEDGE-typed evidence
and is safe to cite via `[evidence: tool_pathway_lookup_<digest>]`.

| Parameter  | Type        | Default      | Description                                                                                  |
| ---------- | ----------- | ------------ | -------------------------------------------------------------------------------------------- |
| `genes`    | array<str>  | -            | Gene symbols to look up. Mixed case is fine. Required.                                       |
| `source`   | enum        | `progeny`    | One of `progeny` / `dorothea` / `collectri`.                                                 |
| `organism` | str         | `human`      | Organism the network is curated for; `human` and `mouse` are supported by all three sources. |
| `top_n`    | int         | `25`         | Cap on regulators / matches surfaced in the response.                                        |

## Reasoning

### `chains`

Submit (or update) a `MechanismChain` candidate and persist it as
`status="proposed"` in `ChainStore`.

| Parameter              | Type            | Description                                                                                                                                  |
| ---------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `summary`              | str             | Required. Short summary of the chain.                                                                                                        |
| `evidence_ids`         | array<str>      | Required. Evidence ids that support the chain. Must be ids actually returned by tool calls (e.g. `tool_scanpy_de_<digest>`).                 |
| `links`                | array<object>   | Optional. Each link has `from_node`, `to_node`, `relation`, `evidence_ids`.                                                                  |
| `chain_id`             | str             | Optional id when updating a previously created chain.                                                                                        |
| `verification_readout` | str             | Optional suggested experimental readout.                                                                                                     |
| `priority`             | enum            | Optional `P1` / `P2` / `P3`. If omitted, defaults to `P1` when `len(evidence_ids) >= 2` else `P2`.                                           |

### `chain_status`

Transition a previously proposed chain. Auto-appends a one-line entry
`<chain_id> -> <status>: <summary>` to `HYPOTHESIS_LOG.md` so the latest
state is visible in the prompt at the start of the next session.

| Parameter      | Type        | Description                                                                                          |
| -------------- | ----------- | ---------------------------------------------------------------------------------------------------- |
| `chain_id`     | str         | Required. Id returned by `chains`.                                                                   |
| `status`       | enum        | Required. `proposed` / `supported` / `refuted` / `superseded`.                                       |
| `evidence_ids` | array<str>  | Optional new evidence ids supporting the transition (merged with the chain's existing evidence).     |
| `note`         | str         | Optional free-text note (e.g. wet-lab readout).                                                      |

## Memory & skills

### `evidence`

Render a recap of the in-process evidence store -- mostly used by the
agent itself before composing a chain.

| Parameter | Type | Description                                              |
| --------- | ---- | -------------------------------------------------------- |
| (none)    | -    | The tool reads from `AgentLoop._evidence_store`.         |

### `evidence_search`

Cross-session search backed by SQLite + FTS5.

| Parameter   | Type   | Description                                                                                                            |
| ----------- | ------ | ---------------------------------------------------------------------------------------------------------------------- |
| `query`     | str    | Free-text FTS5 query against `summary / genes / pathways / source / tool_name`.                                        |
| `gene`      | str    | Substring filter on the `genes_json` column.                                                                           |
| `pathway`   | str    | Substring filter on the `pathways_json` column.                                                                        |
| `tissue`    | str    | Substring filter on `state_conditions / source / summary` (e.g. any tissue / disease term that appears in those fields). |
| `tool_name` | str    | Exact match on the producing tool, e.g. `scanpy_de`.                                                                   |
| `top_k`     | int    | Default `20`. Maximum entries to return.                                                                               |

### `memory`

Mutate one of the three memory targets (`context`, `researcher`,
`hypothesis_log`).

| Parameter   | Type | Description                                                                                                  |
| ----------- | ---- | ------------------------------------------------------------------------------------------------------------ |
| `action`    | enum | `add` / `replace` / `remove`.                                                                                |
| `target`    | enum | `context` / `researcher` / `hypothesis_log`.                                                                 |
| `content`   | str  | New entry text (`add` / `replace`).                                                                          |
| `old_text`  | str  | Unique substring of the existing entry to replace or remove.                                                 |

### `skills_list` / `skill_view` / `skill_manage`

Inspect and mutate the procedural memory under `~/.cytopert/skills/`.

- `skills_list` -- no parameters; returns Level-0 metadata of every live skill.
- `skill_view(name)` -- read the full SKILL.md body.
- `skill_manage(action, name, ...)` -- the writer surface used by the
  reflection module to stage agent-proposed skills. `action` is one of
  `create`, `patch`, `edit`, `delete`, `accept_staged`, `view_file`,
  `write_file`. See `cytopert/skills/tool.py` for the per-action
  parameter list.
