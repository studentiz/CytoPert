# Quick Start

## Install

```bash
git clone <repo>
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

This creates `~/.cytopert/config.json` and a workspace directory.

## Configure LLM

**DashScope (OpenAI‑compatible)**

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
      "model": "qwen3-max"
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
      "model": "anthropic/claude-sonnet-4-20250514"
    }
  }
}
```

## Use the Agent

Single message:

```bash
cytopert agent -m "What tools can I use to analyze perturbation data?"
```

Interactive:

```bash
cytopert agent
```

Interactive commands:
- `/reset` – clear history
- `/exit` or `/quit` – exit

## Run a Workflow

```bash
cytopert run-workflow nfatc1_mammary
```

With feedback:

```bash
cytopert run-workflow nfatc1_mammary --feedback "Experiment X refuted chain A."
```
