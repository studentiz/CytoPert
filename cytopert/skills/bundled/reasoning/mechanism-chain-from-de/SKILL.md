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
1. Call `evidence` to refresh the evidence summary; identify entries: `de_*`, `path_*` (or any earlier `pert_*`).
2. Compose links upstream → downstream:
   - `(perturbation gene) -> (TF / pathway from enrichment)` with relation `regulates` / `activates` / `represses`.
   - `(pathway) -> (state-specific readout, e.g. luminal differentiation)` with relation `drives` / `biases`.
3. Call `chains summary="..." links=[...] evidence_ids=["de_*","path_*"]`. Record the returned chain id.
4. Persist the lifecycle: `chain_status chain_id=<id> status=proposed evidence_ids=[...] note="initial draft"`.
5. Append a one-line summary to `memory(action='add', target='hypothesis_log', content='<chain_id> <gene>->...->...  (proposed)')` so it survives across sessions.

## Pitfalls
- Never invent evidence IDs; only cite those returned by tools or visible in `evidence` / `evidence_search`.
- Chains with fewer than 2 evidence entries default to priority P2 — that's intentional, don't over-claim P1.
- When wet-lab feedback arrives, update via `chain_status` to `supported` / `refuted`; never silently delete the chain.

## Verification
- `cytopert chains show <id>` lists the create + status_change events with non-empty evidence_ids.
- `~/.cytopert/memory/HYPOTHESIS_LOG.md` contains the new chain id within character budget.
