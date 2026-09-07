# Replay provider matrices

PromptWitness can run several named providers over the same rendered scenario
matrix and compare their successful output digests without storing provider
payloads in the report. `ProviderMatrix` is provider-neutral: a provider is any
callable accepted by `ScenarioExecutor`. The checked-in CLI uses deterministic
replay providers so a matrix can be reviewed without network access.

## Configuration

```json
{
  "format": "promptwitness.replay-providers.v1",
  "providers": [
    {
      "name": "baseline",
      "responses": {"<prompt-digest>": {"text": "recorded output"}},
      "retries": 1
    }
  ]
}
```

Response keys are rendered prompt digests, not scenario IDs. This prevents a
fixture from silently matching a changed prompt. Unknown fields, duplicate
provider names, malformed response maps, and invalid retry counts fail before
execution.

## CLI

```console
promptwitness provider-matrix prompt.json scenarios.json providers.json \
  --workers 2 --scenario-workers 2 --output matrix-report.json
```

The report retains scenario IDs, prompt/output digests, attempts, and errors;
it deliberately omits raw provider outputs. `pairwise_agreement` is the
fraction of comparable successful provider pairs with identical output
digests. Exit status is zero only when every provider completed every scenario.
Use `--allow-missing` to record replay misses instead of failing the render
step; provider failures remain visible in the report.

## Python API

```python
from promptwitness import ProviderMatrix, ProviderSpec, load_replay_providers

matrix = ProviderMatrix(load_replay_providers("providers.json"))
report = matrix.run(document, scenarios, workers=2)
print(report.pairwise_agreement)
```

`workers` parallelizes provider runs while `scenario_workers` controls the
ordered executor inside each provider. The returned run order always matches
the declared provider order.

