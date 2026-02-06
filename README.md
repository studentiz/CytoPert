<div align="center">

# 🧬 CytoPert

**Evidence‑bound, tool‑driven framework for reasoning about differential cell‑state responses to perturbations.**

![Status](https://img.shields.io/badge/status-alpha-orange)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CLI](https://img.shields.io/badge/interface-CLI-6a5acd)

</div>

## ✨ What Is CytoPert

CytoPert is an interactive framework for **cell perturbation differential response mechanism parsing**. It helps researchers identify trigger state conditions, decisive regulatory nodes, and produce mechanism chains that can be supported or refuted by experiments.

## 🔬 Why It Matters

The same perturbation can produce different outcomes across cell states. CytoPert organizes **evidence‑bound reasoning** and provides a **plan‑before‑execute** workflow so hypotheses and experiments form a closed loop.

## ✅ Highlights

- 🧠 Evidence‑bound reasoning with traceable citations
- 🧭 Plan‑before‑execute agent workflow
- 🧰 Tool‑driven analysis with scanpy, pertpy, decoupler, and cellxgene Census
- 🧑‍💻 Interactive CLI for single‑turn and multi‑turn sessions

## 🗂 Project Structure

- `cytopert/agent/` – agent loop, context, tool registry
- `cytopert/data/` – Census client, evidence models
- `cytopert/knowledge/` – pathway/topology constraints (stubs by default)
- `cytopert/workflow/` – pipeline and scenarios (e.g. `nfatc1_mammary`)
- `cytopert/providers/` – LLM provider (LiteLLM)
- `cytopert/cli/` – CLI commands

## 🧱 Requirements

- Python 3.11+
- Recommended: conda environment

## 📦 Installation

```bash
git clone <repo>
cd CytoPert
pip install -e .
```

Optional dev tools:

```bash
pip install -e .[dev]
```

## 🚀 Quick Start

### 1) Initialize

```bash
cytopert onboard
```

This creates `~/.cytopert/config.json` and a workspace directory.

### 2) Configure LLM

CytoPert uses LiteLLM and supports OpenAI‑compatible endpoints.

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

### 3) Chat with the Agent

Command | Purpose
---|---
`cytopert agent -m "..."` | single message
`cytopert agent` | interactive session

Interactive commands:
- `/reset` – clear session history
- `/exit` or `/quit` – exit

### 4) Run a Workflow Scenario

```bash
cytopert run-workflow nfatc1_mammary
```

With experiment feedback:

```bash
cytopert run-workflow nfatc1_mammary --feedback "Experiment X refuted chain A."
```

## 📚 Data Access

CytoPert can load data from cellxgene Census or local `.h5ad` files.

### Census Query Tool

The agent can call `census_query` to retrieve a data slice.

Common parameters:
- `obs_value_filter` – SOMA filter for cells
- `var_value_filter` – SOMA filter for genes
- `census_version` – recommended for reproducibility (e.g. `2025-11-08`)
- `organism` – default `Homo sapiens`
- `obs_only` – `true` to fetch only metadata (faster)
- `max_cells` – default `20000`
- `timeout_seconds` – default `30`

Example (metadata only):

```text
census_query obs_value_filter="tissue_ontology_term_id == 'UBERON:0000178'"
obs_only=true max_cells=20000 timeout_seconds=60 census_version=2025-11-08
```

### Local h5ad

```text
load_local_h5ad path="/path/to/your/data.h5ad"
```

## 🧩 Docs

- `docs/README.md`
- `docs/overview.md`
- `docs/quickstart.md`
- `docs/tools.md`
- `docs/workflows.md`
- `docs/troubleshooting.md`

## 🧪 Tests

```bash
pytest -q
```

## ⚠️ Troubleshooting

- Model returns `OK.` repeatedly: use `/reset` or start a new session ID.
- Census queries timing out: narrow filters, use `obs_only=true`, set `max_cells`, and pass `census_version`.
- Provider errors: verify `apiKey`, `apiBase`, and model name in `~/.cytopert/config.json`.

## 📄 License

MIT.
