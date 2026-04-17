# Hermes Agent — Borrowing Blueprints

This directory tracks **read-only blueprint copies** of selected modules from
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent), used
as design references during the CytoPert completeness overhaul (plan stages
1, 4, 6, 7).

These files are tracked by git so reviewers can diff CytoPert's adaptations
against upstream, but they are **excluded from the built wheel** (see the
`[tool.hatch.build.targets.wheel].exclude` block in `pyproject.toml`).
Importantly, `references/hermes/` is also marked `linguist-vendored=true` in
`.gitattributes` so GitHub language statistics for the CytoPert repo do not
count it.

## License & Attribution

Upstream license: **MIT** — see
[upstream LICENSE](https://github.com/NousResearch/hermes-agent/blob/main/LICENSE).
CytoPert is Apache-2.0; MIT is compatible with downstream Apache-2.0 use.

Each adapted file inside `cytopert/` carries a header of the form:

```
"""...module purpose...

Adapted from NousResearch/hermes-agent
<commit-sha>:<upstream-path> (MIT License).
See docs/hermes-borrowing.md for the per-module diff rationale.
"""
```

The full upstream copyright notice is preserved at the top of every file in
this `references/hermes/` tree (the files are unmodified raw downloads).

## Pinned Commit

- **SHA**: `24342813fe2196335ac8e510e8f59f716197d0e8`
- **Date**: 2026-04-17T11:25:47Z
- **Branch tip**: `main`
- **Subject**: `fix(qqbot): correct Authorization header format in send_message REST path (#11569)`
- **Tree**: `4806cbe5fb6927d2a56067f9c4480d1bcee60c68`
- **Browse**: <https://github.com/NousResearch/hermes-agent/tree/24342813fe2196335ac8e510e8f59f716197d0e8>

To re-pin a newer SHA, edit this README, replace the SHA in the curl block
below, then re-run the curl block. Diff each file in this tree before
committing the bump.

```bash
SHA=24342813fe2196335ac8e510e8f59f716197d0e8
BASE="https://raw.githubusercontent.com/NousResearch/hermes-agent/$SHA"
mkdir -p references/hermes/{tools,agent,hermes_cli}
for path in tools/registry.py agent/context_engine.py agent/context_compressor.py \
            agent/prompt_caching.py agent/prompt_builder.py agent/trajectory.py \
            hermes_cli/plugins.py hermes_state.py; do
  curl -fsSL "$BASE/$path" -o "references/hermes/$path"
done
```

## Borrowing Map

The CytoPert column lists the file we will produce; the *Strategy* column
records how we plan to derive it (per-module diff rationale lives in
[docs/hermes-borrowing.md](../../docs/hermes-borrowing.md), produced in plan
stage 3).

| Hermes file | Lines | CytoPert target | Strategy | Plan stage |
| --- | --- | --- | --- | --- |
| `tools/registry.py` | 482 | `cytopert/agent/tools/registry.py` | Adapt pattern (~200 lines): self-register + RLock + `tool_error`/`tool_result` helpers; drop MCP & budget helpers | 1.B |
| `agent/context_engine.py` | 184 | `cytopert/agent/context_engine.py` | Adopt ABC nearly verbatim; add `evidence_id_protect` field for CytoPert | 4.B |
| `agent/context_compressor.py` | 1163 | `cytopert/agent/context_compressor.py` | Borrow *idea only* (~200 lines): protect-first/last + summarize-middle, no LCM/DAG | 4.C |
| `agent/prompt_caching.py` | 72 | `cytopert/providers/prompt_caching.py` | Port near-verbatim (~50 lines): Anthropic system_and_3 cache markers | 6.B |
| `agent/prompt_builder.py` | 1045 | rewrite of `cytopert/agent/context.py` | Borrow *block structure*: identity + memory + skills + context_files + tool_guidance + model_specific | 4 (also informs 2) |
| `agent/trajectory.py` | 56 | `cytopert/agent/trajectory.py` | Port verbatim + add CytoPert metadata fields (evidence_ids, chains_touched) | 7.4 |
| `hermes_cli/plugins.py` | 843 | `cytopert/plugins/manager.py` | Borrow *three-source discovery* (~250 lines): user / project / entry_points; drop lifecycle hooks & callbacks | 7.3 |
| `hermes_state.py` | 1238 | enhancements to `cytopert/persistence/schema.py` | Borrow *schema patterns*: FTS5 triggers, lineage tracking, write-contention handling | 2 (foundation) |

## What we explicitly do NOT borrow

- `gateway/`, `acp_adapter/`, `cron/`, `optional-skills/`, `tinker-atropos/` —
  not needed for single-cell research.
- Three API modes (chat_completions / codex_responses / anthropic_messages)
  — CytoPert uses LiteLLM's unified abstraction.
- Subagent delegation, MCP, browser tools, terminal backends, Honcho user
  modeling, SOUL.md personalities — out of scope for CytoPert.
