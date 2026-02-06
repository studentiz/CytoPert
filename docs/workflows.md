# Workflows

CytoPert workflows encapsulate analysis scenarios. A workflow typically:

1. Builds a plan (tools + data)
2. Loads data (Census or local h5ad)
3. Runs analysis (scanpy/pertpy/decoupler)
4. Collects evidence
5. Produces mechanism chains

## `nfatc1_mammary`

Example scenario for mammary development perturbation:

```bash
cytopert run-workflow nfatc1_mammary
```

Optional:

```bash
cytopert run-workflow nfatc1_mammary --feedback "Experiment X refuted chain A."
```

To add new scenarios, extend `cytopert/workflow/scenarios/` and register in the pipeline.
