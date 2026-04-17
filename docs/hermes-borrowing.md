# Borrowing from hermes-agent

CytoPert's agent loop and learning-loop machinery are adapted from
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
(MIT License). This document records, per module, what we borrowed and
what we changed. The `references/hermes/` tree carries the full upstream
copies at a pinned commit so reviewers can run a real diff.

## Pinned upstream commit

- **SHA**: `24342813fe2196335ac8e510e8f59f716197d0e8` (2026-04-17,
  `fix(qqbot): correct Authorization header format ...`).
- **Browse**: <https://github.com/NousResearch/hermes-agent/tree/24342813fe2196335ac8e510e8f59f716197d0e8>
- See [`references/hermes/README.md`](../references/hermes/README.md) for
  the curl-block that re-pins newer commits.

## Borrowing scope

| Hermes file                      | Lines  | CytoPert target                                        | Strategy                                                       | Plan stage   |
| -------------------------------- | ------ | ------------------------------------------------------ | -------------------------------------------------------------- | ------------ |
| `tools/registry.py`              | 482    | `cytopert/agent/tools/registry.py`                     | Pattern adapted (~360 lines): RLock, ToolEntry, helpers        | 1.B (done)   |
| `agent/context_engine.py`        | 184    | `cytopert/agent/context_engine.py`                     | ABC adopted near-verbatim, plus `evidence_id_protect`           | 4.B (pending)|
| `agent/context_compressor.py`    | 1163   | `cytopert/agent/context_compressor.py`                 | Idea borrowed (~200 lines): protect-first/last + summarize-mid | 4.C (pending)|
| `agent/prompt_caching.py`        | 72     | `cytopert/providers/prompt_caching.py`                 | Near-verbatim port (~50 lines)                                 | 6.B (pending)|
| `agent/prompt_builder.py`        | 1045   | rewrite of `cytopert/agent/context.py`                 | Block-structure borrowed                                       | 4 (informs)  |
| `agent/trajectory.py`            | 56     | `cytopert/agent/trajectory.py`                         | Verbatim port + CytoPert-specific metadata                     | 7.4 (pending)|
| `hermes_cli/plugins.py`          | 843    | `cytopert/plugins/manager.py`                          | Three-source discovery borrowed (~250 lines)                   | 7.3 (pending)|
| `hermes_state.py`                | 1238   | enhancements to `cytopert/persistence/schema.py`       | Schema patterns: FTS5 triggers, lineage, contention            | informs 2/8  |

## Per-module diff rationale (kept current)

### `tools/registry.py` (stage 1.B, done)

Borrowed:

- Module-level `registry` singleton.
- AST-gated `discover_self_registering_tools` for plugin packages.
- RLock around mutation and read paths so plugin reloads cannot race
  with agent dispatch.
- `tool_error` / `tool_result` JSON helpers.
- "Reject shadowing instead of silently overwriting" registration rule.

Dropped (intentional):

- The MCP toolset alias machinery -- CytoPert does not speak MCP.
- Per-tool `max_result_size_chars` budget; CytoPert relies on the
  upcoming ContextCompressor (stage 4.C) instead.
- Synchronous dispatch with `_run_async`; CytoPert's `AgentLoop` is
  fully async, so dispatch stays an `async` coroutine.

Added (CytoPert-specific):

- Backward-compatible `register(tool: Tool)` that takes the existing
  `cytopert.agent.tools.base.Tool` ABC.
- `register_function(...)` for plugin packages that want hermes-style
  function-based registration.
- File-level "Adapted from" header pointing at the pinned upstream
  commit.

### Future modules

Each future stage that adapts a hermes module will append its own
section here when it lands. The expected entries (stages 4 / 6 / 7) are
sketched in the table above so reviewers can predict the diff surface.

## License obligations

- `references/hermes/*` retains the upstream MIT license (the files are
  unmodified raw downloads). They are tracked in git for diffability but
  excluded from the built wheel via the
  `[tool.hatch.build.targets.wheel].exclude` block in `pyproject.toml`.
- Each adapted file inside `cytopert/` carries an `Adapted from
  NousResearch/hermes-agent <sha>:<path> (MIT License)` header.
- CytoPert remains Apache-2.0; downstream MIT use is compatible.
