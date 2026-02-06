# Overview

CytoPert is an interactive framework for **cell perturbation differential response mechanism parsing**. It helps researchers identify trigger state conditions, decisive regulatory nodes, and output mechanism chains that can be supported or refuted by experiments.

## Design Principles

- **Evidence‑bound reasoning**: every conclusion must reference traceable evidence.
- **Plan‑before‑execute**: the agent should outline an analysis plan before heavy computation.
- **Tool‑driven analysis**: scverse tools and Census queries are invoked as tools.
- **Extensible**: new scenarios and constraints can be plugged in.

## Architecture (High Level)

- **Agent loop**: builds context, calls LLM, executes tools, collects evidence.
- **Tools**: Census query, scanpy preprocessing, perturbation analysis, enrichment, mechanism chains.
- **Evidence**: standardized evidence models and summaries.
- **Workflows**: scenario‑based pipelines (e.g. `nfatc1_mammary`).

## Core Folders

- `cytopert/agent/` – agent loop, context builder, tool registry
- `cytopert/data/` – census client, evidence models
- `cytopert/knowledge/` – pathway/topology constraints (stubs by default)
- `cytopert/workflow/` – scenarios and pipeline
- `cytopert/providers/` – LLM provider integrations
