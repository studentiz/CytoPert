---
name: mechanism-chain-from-de
description: Convert top DE genes + enrichment into a MechanismChain candidate with verification readout
version: 0.1.0
metadata:
  cytopert:
    category: reasoning
    tags: [chains, mechanism, evidence]
    requires_tools: [chains, chain_status, evidence]
---

# Mechanism Chain From DE

## When to Use
- You have at least one DE evidence entry AND one enrichment evidence entry produced in the current session OR retrievable via `evidence_search`.
- Researcher asks for a mechanism chain or you completed a `perturbation-de` skill run.

## Procedure
1. Call `evidence` to refresh the evidence summary; identify entries by their real ids returned by previous tool calls. Live ids follow the `tool_<tool_name>_<digest>` scheme (e.g. `tool_scanpy_de_3a4b5c6d7e`); never invent a `de_*` / `path_*` prefix.
2. Compose links upstream → downstream:
   - `(perturbation gene) -> (TF / pathway from enrichment)` with relation `regulates` / `activates` / `represses`.
   - `(pathway) -> (state-specific readout)` with relation `drives` / `biases`. The state readout can be any cell-state phenotype the researcher is investigating (differentiation outcome, activation marker shift, trajectory branch choice, ...) — adapt to the dataset, do not hard-code a specific tissue.
3. Call `chains summary="..." links=[...] evidence_ids=["tool_scanpy_de_..."]`. Record the returned chain id.
4. Persist the lifecycle: `chain_status chain_id=<id> status=proposed evidence_ids=[...] note="initial draft"`. The chain_status tool now also auto-appends a one-line entry to `HYPOTHESIS_LOG.md` so you do not need a separate `memory.add` call for the lifecycle line.

## Pitfalls
- Never invent evidence IDs; only cite those returned by tools or visible in `evidence` / `evidence_search`. The `tool_<tool_name>_<digest>` shape is required.
- Chains with fewer than 2 evidence entries default to priority P2 — that's intentional, don't over-claim P1. If you really need a different priority, pass `priority` explicitly to the `chains` tool.
- When wet-lab feedback arrives, update via `chain_status` to `supported` / `refuted`; never silently delete the chain.

## Verification
- `cytopert chains show <id>` lists the create + status_change events with non-empty evidence_ids.
- `~/.cytopert/memory/HYPOTHESIS_LOG.md` contains a one-line entry of the form `chain_<id> -> <status>: <summary>` (auto-written by chain_status).
