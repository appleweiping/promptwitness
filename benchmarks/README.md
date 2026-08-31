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
