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

Interactive commands:

- `/reset` -- clear history
- `/exit` or `/quit` -- exit

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

The pluggable `Pipeline` / `Stage` registry that lets third-party
packages add their own scenarios is scheduled for stage 7.1; the CLI
currently only routes `nfatc1_mammary` through the legacy single-stage
pipeline.
