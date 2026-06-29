# Cenbench Docker image

This benchmark is available as a Docker image at:
`ghcr.io/robinlovelace/cenbench:latest`

## Usage in tdscience/tds

Add to `tdscience/tds` repository:

```markdown
## Cenbench

Pedestrian flow benchmarks are containerised at
[ghcr.io/robinlovelace/cenbench](https://github.com/Robinlovelace/cenbench/pkgs/container/cenbench).

```bash
docker pull ghcr.io/robinlovelace/cenbench:latest
docker run --rm ghcr.io/robinlovelace/cenbench:latest
```

See [github.com/Robinlovelace/cenbench](https://github.com/Robinlovelace/cenbench)
for source code, benchmark results, and pipeline documentation.
```
