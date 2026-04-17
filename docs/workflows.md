# Workflows

A workflow scenario wraps an opinionated multi-step analysis around the
agent loop. CytoPert itself is **domain-agnostic** — it does not assume
any particular tissue, organism, perturbation modality, or disease. The
bundled `nfatc1_mammary` is a *reproducible example*, not a default; you
should register your own scenario for any project that does not happen to
be NFATC1 in mammary cells. The pluggable `Stage` / `Pipeline` registry
(stage 7.1) is what makes that registration possible.

A scenario typically:

1. Builds a research question and an optional `data_config` payload
   (cell-type filters, perturbation gene, Census filter expression, ...).
2. Calls `cytopert.workflow.pipeline.run_one_round`, which constructs a
   `LiteLLMProvider` + `AgentLoop` from the active `Config` and invokes
   `AgentLoop.process_direct(prompt, session_key="workflow:one_round")`.
3. The agent-side learning loop persists evidence, chains, and (when the
   reflection thresholds trip) memory updates and staged skills.

`run_one_round` now also threads `agents.defaults.{maxTokens, temperature}`
into the provider call, so config values actually reach the LLM (this was
silently dropped in earlier alphas).

## `nfatc1_mammary` (bundled example)

A reproducible mammary-development perturbation example. **Use it to
sanity-check the install or as a template; do not assume CytoPert is
specific to NFATC1, mammary tissue, or breast biology.** Other
perturbations / tissues / organisms are first-class — just register a
new scenario via `register_scenario(...)`.

```bash
cytopert run-workflow nfatc1_mammary
```

With feedback:

```bash
cytopert run-workflow nfatc1_mammary --feedback "Experiment X refuted chain A."
```

The `--feedback` payload is appended to the research question and is
also forwarded to `AgentLoop.process_direct(..., user_feedback=...)`, so
the reflection module can decide whether to advance a chain to
`supported` / `refuted` even when the message itself does not trigger
the other reflection thresholds (`>=5 tool calls`, chain touched,
`>=3 new evidence entries`).

## Adding scenarios (today, before stage 7.1)

1. Drop a new module under `cytopert/workflow/scenarios/` exposing a
   `run(research_question, scenario_config=None, feedback=None)` function
   that returns `{"response": str, "scenario": str, "config": dict}`.
2. Add an `if scenario == "<name>": ...` branch to
   `cytopert/cli/commands.py:run_workflow`. (Stage 7.1 replaces this with
   a `SCENARIO_REGISTRY` lookup that auto-discovers same-package modules
   plus `cytopert.scenarios` entry-points.)
3. Surface defaults under `Config.workflow["<name>"]` so `cytopert
   run-workflow <name>` can merge user overrides via
   `cytopert.workflow.pipeline.get_scenario_config`.

## Roadmap

- The `Stage` / `Pipeline` dataclasses and the `SCENARIO_REGISTRY`
  populated from `cytopert/workflow/scenarios/*.py` plus
  `importlib.metadata.entry_points(group="cytopert.scenarios")` shipped
  in stage 7.1; new scenarios just need to call `register_scenario(...)`
  in their module top-level.
- The `PlanGate` state machine ships in the AgentLoop today but is
  **opt-in** (`/plan-gate on` in the interactive shell) -- the previous
  default-on policy was rolled back because it forced every casual
  message into a plan-only turn that ignored the user's actual request.
  Workflow scenarios that want plan-then-confirm semantics should call
  `agent_loop.enable_plan_gate(session_key)` before invoking
  `Pipeline.run`.
