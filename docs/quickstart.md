# Quick Start

## Install

```bash
git clone https://github.com/your-org/CytoPert.git
cd CytoPert
pip install -e .
```

Optional dev tools:

```bash
pip install -e .[dev]
```

## Initialize

```bash
cytopert onboard
```

`onboard` creates `~/.cytopert/` with:

- `config.json` -- provider keys, model defaults
- `workspace/` -- intermediate `.h5ad` outputs from scanpy tools
- `memory/{CONTEXT,RESEARCHER,HYPOTHESIS_LOG}.md` -- agent-curated notes
- `chains/` -- per-chain JSONL audit trails
- `skills/` -- bundled `SKILL.md` sheets, plus the `.staged/` quarantine
  for agent-proposed skills

`state.db` (SQLite + FTS5 evidence + chains) and `sessions/` are created
lazily on first use, not at onboard time. `CYTOPERT_HOME` overrides the
root, which is useful for one-isolated-state-per-project.

## Configure LLM

**DashScope (OpenAI-compatible)**

```json
{
  "providers": {
    "openai": {
      "apiKey": "YOUR_API_KEY",
      "apiBase": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }
  },
  "agents": {
    "defaults": {
      "model": "qwen3-max",
      "maxTokens": 8192,
      "temperature": 0.3
    }
  }
}
```

**OpenRouter**

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-xxx",
      "apiBase": "https://openrouter.ai/api/v1"
    }
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-sonnet-4-20250514",
      "maxTokens": 8192,
      "temperature": 0.3
    }
  }
}
```

`maxTokens` and `temperature` from `agents.defaults` are now actually
threaded into every `provider.chat` call (fixed in stage 2 of the
completeness overhaul). Run `cytopert status` to verify the resolved
provider, model, and API-key state.

## Use the Agent

Single message:

```bash
cytopert agent -m "What tools can I use to analyze perturbation data?"
```

Single message with wet-lab feedback (forwarded to the reflection
module so chains can be moved through the lifecycle):

```bash
cytopert agent -s mammary_nfatc1 -f "qPCR n=6 p=0.42 -- mark chain_0001 refuted."
```

Interactive:

```bash
cytopert agent
```

Interactive commands (type `/help` to print the full list):

- `/help`             -- show every slash command
- `/reset` or `/new`  -- clear conversation, re-arm the plan gate
- `/skip-plan`        -- disable plan-gate for this session
- `/model [name]`     -- switch model in-process
- `/usage`            -- print this session's tokens / cost
- `/history [N]`      -- print the last N user / assistant messages
- `/skills`           -- list installed skills
- `/chains`           -- list recent mechanism chains
- `/retry`            -- re-send the last user message
- `/undo`             -- drop the last user + assistant turn
- `/exit` or `/quit`  -- exit

## Inspect persistent state

```bash
cytopert chains list                   # rich table; one row per chain
cytopert chains show chain_0001        # links + lifecycle events
cytopert evidence search BRCA1         # cross-session FTS5
cytopert memory show -t hypothesis_log # latest chain transitions
cytopert skills list --include-staged  # plus agent-proposed skills
```

## Run a Workflow

```bash
cytopert run-workflow nfatc1_mammary
```

With feedback:

```bash
cytopert run-workflow nfatc1_mammary --feedback "Experiment X refuted chain A."
```

`cytopert run-workflow --help` lists every registered scenario name; new
scenarios are loaded automatically from `cytopert.workflow.scenarios`
and from any installed plugin that calls `register_scenario(...)` in
its `setup` hook. See `docs/workflows.md` for the contract.

## Profiles (isolated states)

Each researcher / study can have its own root under
`~/.cytopert/profiles/<name>/` with a complete tree of config, workspace,
memory, chains, skills, sessions and `state.db`. Three ways to switch:

```bash
cytopert profile new study42                    # create
cytopert -p study42 setup                       # initialise it
cytopert -p study42 agent -m "..."              # one-shot override
cytopert profile use study42                    # persist as default
cytopert profile list                           # see all profiles
```

Switching profiles re-routes every command (`agent`, `chains`,
`evidence`, `memory`, `skills`, `cron`, ...) to the selected directory
without touching the default root.

## Skills hub

Beyond the bundled SKILL.md sheets, you can install user / community
skills from local directories, archives, or git URLs:

```bash
cytopert skills search wilcoxon                  # name + description match
cytopert skills install ./my-skill/              # local directory
cytopert skills install ./my-skill.tar.gz        # .zip / .tar.gz / .tgz
cytopert skills install https://github.com/owner/repo.git --name xyz
cytopert skills uninstall xyz
```

`install` always lands the skill at `<profile>/skills/<category>/<name>/`
so multiple profiles can hold disjoint skill libraries.

## Cron-style scheduling

`cytopert cron` lets you queue recurring agent runs without leaning on
the host's cron / launchd:

```bash
cytopert cron add "every 30m" -m "Refresh CONTEXT.md from the latest evidence."
cytopert cron add "daily" --scenario generic_de --feedback "Weekly QC."
cytopert cron list                       # rich table with next-run timestamps
cytopert cron tick --dry-run             # show what would run, do not run it
cytopert cron tick                       # execute every due job once and exit
cytopert cron daemon --interval 60       # run forever in 60s ticks (Ctrl+C to stop)
cytopert cron disable myjob              # keep the entry but skip it
cytopert cron remove myjob               # drop the entry
```

The schedule grammar is intentionally tiny: `every Ns/Nm/Nh/Nd`, plus the
shorthand aliases `minutely` / `hourly` / `daily`. Job state lives at
`<profile>/jobs.json`, so each profile owns an independent schedule.
