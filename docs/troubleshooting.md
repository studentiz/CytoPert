# Troubleshooting

## Model responds with `OK.` repeatedly

Cause: session history contains a short‑reply pattern.

Fix:
- In interactive mode, run `/reset`
- Or start a new session id: `cytopert agent -s cli:new`

## Census queries time out

Fixes:
- Add narrower `obs_value_filter`
- Use `obs_only=true`
- Set `max_cells` (default 20000)
- Provide `census_version`

## LLM connection errors

Fixes:
- Verify `apiKey`, `apiBase`, and model name in `~/.cytopert/config.json`
- Retry the request

## No mechanism chains produced

Cause: no evidence available.

Fix:
- Load data via Census or local `.h5ad`
- Run analysis tools to populate evidence
