# <Run Name>

## Run Summary

| Field | Value |
|---|---|
| Run ID | `<run-id>` |
| Status | `<planned, running, completed, failed, or interrupted>` |
| Started | `<ISO 8601 timestamp or not recorded>` |
| Finished | `<ISO 8601 timestamp or not recorded>` |
| Baseline | `<run ID, report path, or not specified>` |
| Git revision | `<commit or not recorded>` |
| Command | `<exact command or not recorded>` |
| Seed | `<seed or not recorded>` |
| Hardware | `<device and host summary or not recorded>` |

## Configuration

Source: `<config path or not recorded>`

```yaml
# Complete effective configuration, including defaults and overrides.
```

## Changed Hyperparameters

| Parameter | Baseline | Current | Source |
|---|---:|---:|---|
| `<name>` | `<value>` | `<value>` | `<file, output, or URL>` |

Use `No confirmed changes` when no explicit baseline is available or no
differences are present.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| `<split>` | `<metric>` | `<value>` | `<step>` | `<min or max>` | `<file, output, or URL>` |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `<path>` | `<yes or no>` |
| Checkpoint | `<path>` | `<yes or no>` |
| Tracking run | `<URL or not recorded>` | `<yes or no>` |
| Plots / report | `<path or URL>` | `<yes or no>` |

## Failure Details

- Last known step: `<step or not applicable>`
- Error summary: `<concise message or not applicable>`
- Detailed output: `<path or not recorded>`

## Observations

- `<Directly supported observation>`

## Interpretation

- `<Clearly labeled hypothesis, limitation, or recommendation>`

## Updates

- `<ISO 8601 timestamp>: <what changed and evidence source>`
