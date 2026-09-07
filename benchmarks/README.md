# PromptWitness benchmarks

`benchmark_diff.py` measures positional and smart alignment on a deterministic
synthetic prompt with one insertion and sparse content edits. It emits environment,
workload, median timing, and result counts as JSON.

```bash
python benchmarks/benchmark_diff.py --messages 250 --repeats 3
```

Smart alignment is quadratic in message count. The benchmark makes that cost visible;
it is not a portable performance guarantee. Compare medians only with identical
arguments and an otherwise idle machine.

`benchmark_fixture.py` renders the checked-in prompt fixture and exercises the
provider-boundary executor with deterministic local scenarios. It records the
fixture digest, environment, timing, and traced memory; this is fixture-real
evidence and is kept separate from generated alignment workloads.

```bash
python benchmarks/benchmark_fixture.py
```
