"""Are single-token edits distinguishable from numerical re-rolls of greedy decoding?

Exploratory analysis, specified after the descriptive re-roll numbers of ``reroll_study.py`` were
seen. Per pool (model x task), from the deterministic-reader outputs in ``reroll3/``:

* edits: ``per_edit[*].correct`` of ``<model>-<task>-token.json``, one 0/1 vector over the held-out
  questions per GReaTer-style single-token edit (read in one request of all questions);
* re-rolls: ``reroll_reads[*].correct`` of ``<model>-<task>-token-rerolls.json`` (the unchanged
  incumbent read in separate requests of c questions) plus the base read ``base_correct``, which is
  itself one draw (one request of all questions). The base read must be identical in both files.

Exchangeability of the two label sets is tested with permutation tests over all relabelings when
there are at most ``--max-enumerate`` of them, else over ``--n-random`` random relabelings with a
fixed seed (p = (1 + count) / (1 + draws)). Statistics: (a) difference of mean accuracies (edits
minus re-rolls); (b) log ratio of the accuracy variances (edits over re-rolls); (c) difference of
the mean fraction of questions whose correctness differs from the base read (edits vs the re-rolls
without the base). Two-sided p = min(1, 2 min(p_upper, p_lower)), which does not depend on monotone
transformations of the statistic (so not on the variance's degrees of freedom). Pooled tests sum
the per-pool statistics standardized by the mean and sd of their permutation null, under
independent relabeling within each pool (``--n-random`` joint draws). A sensitivity variant leaves
the base read out of the re-roll set.

Also reported: accuracy mean and sd (population sd, as in the results digest; sample sd in the
JSON), per-question flip statistics (flips, gains, losses relative to the base read; mean pairwise
disagreement within each set and the sd it implies if every disagreement were a fair coin for the
sign: sqrt(d / (2 n))), and the reasoning-change fields recorded by ``reroll_study.py``.

Standard library only (runs under the repo venv); ``--figure`` needs matplotlib.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from array import array
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

DEFAULT_DIR = Path(r"D:\Company\research-artifacts\graft-runs\reroll3")
MODEL_NAMES = {"llama3": "Llama-3-8B", "qwen3": "Qwen3-8B"}
MODEL_SHORT = {"llama3": "Llama-3", "qwen3": "Qwen3"}
MODEL_INITIAL = {"llama3": "L", "qwen3": "Q"}
TASK_NAMES = {"logical_deduction_seven_objects": "LD7",
              "tracking_shuffled_objects_seven_objects": "TS7"}


def pool_label(model: str, task: str) -> str:
    """Pool label in the style of the paper's Table 1: task, then model initial (e.g. LD7-L)."""
    return f"{TASK_NAMES.get(task, task)}-{MODEL_INITIAL.get(model, model)}"


def record_label(pool: dict) -> str:
    """Label of an analysis record, recomputed so that older JSONs render in the current style."""
    model = {name: key for key, name in MODEL_NAMES.items()}.get(pool["model"])
    return pool_label(model, pool["task"]) if model else pool["pool"]


POOLS = tuple((model, task, pool_label(model, task)) for model, task in (  # (prefix, task, label)
    ("llama3", "logical_deduction_seven_objects"),
    ("llama3", "tracking_shuffled_objects_seven_objects"),
    ("qwen3", "logical_deduction_seven_objects"),
    ("qwen3", "tracking_shuffled_objects_seven_objects"),
))
STATISTICS = ("mean_difference", "log_variance_ratio")
TOL = 1e-9

# A statistic maps (sum_a, sumsq_a, n_a, sum_b, sumsq_b, n_b) of the two groups to a float.
Statistic = Callable[[float, float, int, float, float, int], float]


def _variance(total: float, squares: float, n: int) -> float:
    """Sample variance from the sum and the sum of squares (exact numerator for integers)."""
    return (n * squares - total * total) / (n * (n - 1))


def mean_difference(sa: float, qa: float, na: int, sb: float, qb: float, nb: int) -> float:
    return sa / na - sb / nb


def log_variance_ratio(sa: float, qa: float, na: int, sb: float, qb: float, nb: int) -> float:
    va, vb = _variance(sa, qa, na), _variance(sb, qb, nb)
    if va <= 0 and vb <= 0:
        return 0.0
    if vb <= 0:
        return math.inf
    if va <= 0:
        return -math.inf
    return math.log(va / vb)


STATISTIC_FUNCTIONS: dict[str, Statistic] = {"mean_difference": mean_difference,
                                             "log_variance_ratio": log_variance_ratio}


def _ge(x: float, y: float) -> bool:
    """x >= y up to floating-point noise (values that are ties in exact arithmetic count)."""
    if math.isinf(y) or math.isinf(x):
        return x >= y
    return x >= y - TOL * (1.0 + abs(y))


@dataclass
class PermutationResult:
    statistic: str
    observed: float
    p_two_sided: float
    p_upper: float  # P(T* >= T_obs)
    p_lower: float  # P(T* <= T_obs)
    exact: bool
    n_relabelings: int
    null_mean: float
    null_sd: float
    n_infinite: int
    finite_min: float
    finite_max: float
    null: array = field(repr=False)
    _standardized: array | None = field(default=None, repr=False)

    def z(self, value: float | None = None) -> float:
        """Standardized statistic; infinite values are clipped to the extreme finite null value."""
        x = self.observed if value is None else value
        if math.isinf(x):
            x = self.finite_max if x > 0 else self.finite_min
        return (x - self.null_mean) / self.null_sd if self.null_sd > 0 else 0.0

    def standardized_null(self) -> array:
        if self._standardized is None:
            self._standardized = array("d", (self.z(v) for v in self.null))
        return self._standardized

    def summary(self) -> dict:
        return {"observed": self.observed, "p_two_sided": self.p_two_sided,
                "p_upper": self.p_upper, "p_lower": self.p_lower, "exact": self.exact,
                "n_relabelings": self.n_relabelings, "null_mean": self.null_mean,
                "null_sd": self.null_sd, "z": self.z(), "n_infinite_relabelings": self.n_infinite}


def permutation_tests(a: Sequence[float], b: Sequence[float], names: Sequence[str] = STATISTICS,
                      max_enumerate: int = 1_000_000, n_random: int = 200_000,
                      seed: int = 0) -> dict[str, PermutationResult]:
    """Permutation tests of exchangeability between the label sets ``a`` and ``b``.

    Every relabeling assigns ``len(b)`` of the pooled values to group b. All C(n, len(b))
    relabelings are enumerated when there are at most ``max_enumerate``; otherwise ``n_random``
    random relabelings are drawn with ``random.Random(seed)``.
    """
    values = list(a) + list(b)
    squares = [v * v for v in values]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("each group needs at least two values")
    total_s, total_q = sum(values), sum(squares)
    functions = [STATISTIC_FUNCTIONS[name] for name in names]

    def stats(sb: float, qb: float) -> list[float]:
        return [f(total_s - sb, total_q - qb, na, sb, qb, nb) for f in functions]

    observed = stats(sum(b), sum(v * v for v in b))
    exact = math.comb(na + nb, nb) <= max_enumerate
    nulls = [array("d") for _ in names]
    upper = [0] * len(names)
    lower = [0] * len(names)
    groups: Iterable[tuple[Sequence[float], Sequence[float]]]
    if exact:
        groups = zip(itertools.combinations(values, nb), itertools.combinations(squares, nb),
                     strict=True)
    else:
        rng = random.Random(seed)
        indices = range(na + nb)
        draws = (rng.sample(indices, nb) for _ in range(n_random))
        groups = (([values[i] for i in d], [squares[i] for i in d]) for d in draws)
    count = 0
    for group_values, group_squares in groups:
        count += 1
        for k, t in enumerate(stats(sum(group_values), sum(group_squares))):
            nulls[k].append(t)
            upper[k] += _ge(t, observed[k])
            lower[k] += _ge(-t, -observed[k])
    results = {}
    for k, name in enumerate(names):
        finite = [v for v in nulls[k] if math.isfinite(v)]
        if exact:
            p_up, p_lo = upper[k] / count, lower[k] / count
        else:
            p_up, p_lo = (1 + upper[k]) / (1 + count), (1 + lower[k]) / (1 + count)
        results[name] = PermutationResult(
            statistic=name, observed=observed[k], p_two_sided=min(1.0, 2 * min(p_up, p_lo)),
            p_upper=p_up, p_lower=p_lo, exact=exact, n_relabelings=count,
            null_mean=statistics.fmean(finite), null_sd=statistics.pstdev(finite),
            n_infinite=count - len(finite), finite_min=min(finite), finite_max=max(finite),
            null=nulls[k])
    return results


def pooled_test(results: Sequence[PermutationResult], n_random: int = 200_000,
                seed: int = 0) -> dict:
    """Sum of per-pool standardized statistics; null by independent relabeling within each pool.

    Each draw takes one relabeling uniformly at random per pool (an entry of the pool's null; for
    enumerated nulls this is a uniform random relabeling).
    """
    observed = math.fsum(r.z() for r in results)  # fsum: identical on every Python version
    standardized = [r.standardized_null() for r in results]
    rng = random.Random(seed)
    upper = lower = 0
    for _ in range(n_random):
        s = math.fsum(z[rng.randrange(len(z))] for z in standardized)
        upper += _ge(s, observed)
        lower += _ge(-s, -observed)
    p_up, p_lo = (1 + upper) / (1 + n_random), (1 + lower) / (1 + n_random)
    return {"observed_sum_z": observed, "per_pool_z": [r.z() for r in results],
            "p_two_sided": min(1.0, 2 * min(p_up, p_lo)), "p_upper": p_up, "p_lower": p_lo,
            "n_joint_relabelings": n_random}


def flip_fraction(vector: Sequence[int], base: Sequence[int]) -> float:
    return sum(x != y for x, y in zip(vector, base, strict=True)) / len(base)


def pairwise_disagreement(vectors: Sequence[Sequence[int]]) -> float:
    return statistics.fmean(flip_fraction(x, y) for x, y in itertools.combinations(vectors, 2))


def _describe(values: Sequence[float]) -> dict:
    return {"mean": statistics.fmean(values), "sd_population": statistics.pstdev(values),
            "sd_sample": statistics.stdev(values), "min": min(values), "max": max(values)}


def _median(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pool(directory: Path, model: str, task: str) -> dict:
    token = _read(directory / f"{model}-{task}-token.json")
    rerolls = _read(directory / f"{model}-{task}-token-rerolls.json")
    if token["base_correct"] != rerolls["base_correct"]:
        raise ValueError(f"{model} {task}: base_correct differs between the token and "
                         "re-roll files")
    return {"token": token, "rerolls": rerolls}


def _close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-12


def analyze_pool(data: dict, label: str, model: str, task: str, settings: dict,
                 seed: int) -> tuple[dict, dict]:
    token, rerolls = data["token"], data["rerolls"]
    base = token["base_correct"]
    n_q = len(base)
    edits = list(token["per_edit"].items())
    reads = rerolls["reroll_reads"]
    edit_vectors = [v["correct"] for _, v in edits]
    roll_vectors = [r["correct"] for r in reads]
    edit_acc = [sum(v) / n_q for v in edit_vectors]
    roll_acc = [sum(v) / n_q for v in roll_vectors]
    base_acc = sum(base) / n_q

    # consistency of the recorded fields with the vectors
    edit_flips = [flip_fraction(v, base) for v in edit_vectors]
    roll_flips = [flip_fraction(v, base) for v in roll_vectors]
    checks = {
        "base_correct_identical": True,  # load_pool raises otherwise
        "base_accuracy_matches_summary": _close(base_acc, token["summary"]["base_accuracy"]),
        "edit_flips_match_recorded": all(_close(f, v["correctness_flipped"])
                                         for f, (_, v) in zip(edit_flips, edits, strict=True)),
        "reroll_flips_match_recorded": all(_close(f, r["correctness_flipped"])
                                           for f, r in zip(roll_flips, reads, strict=True)),
        "edit_deltas_match_recorded": all(_close(a - base_acc, v["accuracy_delta"])
                                          for a, (_, v) in zip(edit_acc, edits, strict=True)),
        "reroll_deltas_match_recorded": all(_close(a - base_acc, r["accuracy_delta"])
                                            for a, r in zip(roll_acc, reads, strict=True)),
    }

    # permutation tests on correct counts (integers: exact sums and variance numerators)
    edit_counts = [sum(v) for v in edit_vectors]
    roll_counts = [sum(v) for v in roll_vectors] + [sum(base)]
    common = {"max_enumerate": settings["max_enumerate"], "n_random": settings["n_random"]}
    accuracy_tests = permutation_tests(edit_counts, roll_counts, seed=seed, **common)
    no_base_tests = permutation_tests(edit_counts, roll_counts[:-1], seed=seed + 1, **common)
    flip_tests = permutation_tests([round(f * n_q) for f in edit_flips],  # flipped-question counts
                                   [round(f * n_q) for f in roll_flips], names=("mean_difference",),
                                   seed=seed + 2, **common)

    def summarize(results: dict[str, PermutationResult]) -> dict:
        out = {name: r.summary() for name, r in results.items()}
        # the mean difference was computed on counts; also report it in accuracy units
        out["mean_difference"]["observed_accuracy"] = results["mean_difference"].observed / n_q
        return out

    def gains(v: Sequence[int]) -> float:
        return sum(x == 1 and y == 0 for x, y in zip(v, base, strict=True)) / n_q

    def losses(v: Sequence[int]) -> float:
        return sum(x == 0 and y == 1 for x, y in zip(v, base, strict=True)) / n_q

    edit_shift = statistics.fmean(abs(a - base_acc) for a in edit_acc)
    roll_shift = statistics.fmean(abs(a - base_acc) for a in roll_acc)
    d_edits = pairwise_disagreement(edit_vectors)
    d_rolls = pairwise_disagreement([*roll_vectors, base])
    flip_test = flip_tests["mean_difference"].summary()
    flip_test["observed_fraction"] = flip_tests["mean_difference"].observed / n_q
    per_question = {
        "flip_fraction_vs_base": {"edits": _describe(edit_flips), "rerolls": _describe(roll_flips),
                                  "test_mean_difference": flip_test},
        "gain_fraction_vs_base": {"edits": statistics.fmean(map(gains, edit_vectors)),
                                  "rerolls": statistics.fmean(map(gains, roll_vectors))},
        "loss_fraction_vs_base": {"edits": statistics.fmean(map(losses, edit_vectors)),
                                  "rerolls": statistics.fmean(map(losses, roll_vectors))},
        "mean_abs_accuracy_change_vs_base": {"edits": edit_shift, "rerolls": roll_shift},
        "abs_accuracy_change_per_flip": {"edits": edit_shift / statistics.fmean(edit_flips),
                                         "rerolls": roll_shift / statistics.fmean(roll_flips)},
        "pairwise_disagreement": {"edits": d_edits, "rerolls_incl_base": d_rolls},
        "sign_symmetric_sd": {"edits": math.sqrt(d_edits / (2 * n_q)),
                              "rerolls_incl_base": math.sqrt(d_rolls / (2 * n_q)),
                              "note": "sample sd of accuracy if each pairwise disagreement had a "
                                      "fair-coin sign: sqrt(d / (2 n)); compare with sd_sample"},
    }
    edit_first = [v["first_divergence_median"] for _, v in edits]
    roll_first = [r["first_divergence_median"] for r in reads]
    reasoning = {
        "reasoning_changed": {"edits": _describe([v["reasoning_changed"] for _, v in edits]),
                              "rerolls": _describe([r["reasoning_changed"] for r in reads])},
        "answer_changed": {"edits": _describe([v["answer_changed"] for _, v in edits]),
                           "rerolls": _describe([r["answer_changed"] for r in reads])},
        "first_divergence_median": {
            "edits_median_of_medians": _median(edit_first),
            "edits_min": min(f for f in edit_first if f is not None),
            "edits_max": max(f for f in edit_first if f is not None),
            "rerolls_median_of_medians": _median(roll_first),
            "rerolls_min": min(f for f in roll_first if f is not None),
            "rerolls_max": max(f for f in roll_first if f is not None),
            "note": "edits: questions where one reasoning is a prefix of the other count at the "
                    "shorter length; re-rolls: such questions are left out (reroll_study.py)"},
        "reroll_truncated": [r.get("truncated") for r in reads],
        "base_truncated": rerolls["summary"].get("base_truncated"),
    }
    out = {
        "pool": label, "model": MODEL_NAMES[model], "model_short": MODEL_SHORT[model],
        "task": task, "n_questions": n_q, "n_edits": len(edits), "n_rerolls": len(reads),
        "reroll_chunks": [r["chunk"] for r in reads], "checks": checks,
        "accuracy": {
            "base": base_acc,
            "edits": dict(zip([name for name, _ in edits], edit_acc, strict=True)),
            "rerolls": [{"chunk": r["chunk"], "accuracy": a}
                        for r, a in zip(reads, roll_acc, strict=True)],
            "edits_summary": _describe(edit_acc),
            "rerolls_incl_base_summary": _describe([*roll_acc, base_acc]),
            "rerolls_excl_base_summary": _describe(roll_acc),
        },
        "tests": summarize(accuracy_tests),
        "tests_rerolls_excl_base": summarize(no_base_tests),
        "per_question": per_question, "reasoning": reasoning,
    }
    nulls = {"accuracy": accuracy_tests, "no_base": no_base_tests, "flips": flip_tests}
    return out, nulls


def analyze(directory: Path, max_enumerate: int, n_random: int, seed: int) -> dict:
    settings = {"max_enumerate": max_enumerate, "n_random": n_random, "seed": seed}
    pools, nulls = [], []
    for k, (model, task, label) in enumerate(POOLS):
        out, null = analyze_pool(load_pool(directory, model, task), label, model, task, settings,
                                 seed + 10 * k)
        pools.append(out)
        nulls.append((model, null))
        print(f"{label}: done", flush=True)
    subsets = {"all": ("llama3", "qwen3"), "llama3": ("llama3",), "qwen3": ("qwen3",)}
    pooled: dict = {}
    for k, (subset, models) in enumerate(subsets.items()):
        chosen = [null for model, null in nulls if model in models]

        def combine(kind: str, name: str, offset: int, chosen: list = chosen) -> dict:
            return pooled_test([c[kind][name] for c in chosen], n_random, seed + offset)

        pooled[subset] = {
            "accuracy": {name: combine("accuracy", name, 100 + k) for name in STATISTICS},
            "accuracy_rerolls_excl_base": {name: combine("no_base", name, 200 + k)
                                           for name in STATISTICS},
            "flip_fraction": combine("flips", "mean_difference", 300 + k),
        }
    return {
        "analysis": "reroll_exchangeability",
        "status": "exploratory (specified after the descriptive re-roll numbers were seen)",
        "inputs": {"directory": str(directory),
                   "files": [f"{m}-{t}-token.json, {m}-{t}-token-rerolls.json"
                             for m, t, _ in POOLS]},
        "settings": {
            **settings, "two_sided": "min(1, 2 min(p_upper, p_lower))",
            "random_p": "(1 + count) / (1 + draws)", "accuracy_sd": "population (ddof 0)",
            "variance_in_statistic": "sample (ddof 1); p-values do not depend on it",
            "groups": "edits = per_edit[*].correct; "
                      "re-rolls = reroll_reads[*].correct + base_correct",
            "pooled": "sum of per-pool z (permutation-null mean and sd), independent relabeling "
                      "within each pool; subsets llama3/qwen3 are exploratory"},
        "pools": pools, "pooled": pooled,
    }


def fmt(value: float, digits: int, scale: int = 1) -> str:
    """Round half up on the shortest decimal representation (0.2025 -> 20.3 at scale 100)."""
    quantum = Decimal(1).scaleb(-digits)
    return str((Decimal(repr(value)) * scale).quantize(quantum, rounding=ROUND_HALF_UP))


def fmt_p(p: float) -> str:
    if p >= 0.995:
        return "1.00"
    if p >= 0.01:
        return fmt(p, 2)
    if p >= 0.001:
        return fmt(p, 3)
    return r"$<$0.001"


def write_latex(result: dict, path: Path) -> None:
    lines = [
        r"\setlength{\tabcolsep}{4pt}% local to the enclosing table environment",
        r"\begin{tabular}{@{}lrrrrccrr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\makecell{Reasoning\\changed (\%)}}"
        r" & \multicolumn{2}{c}{\makecell{Correctness\\flipped (\%)}}"
        r" & \multicolumn{2}{c}{Accuracy (mean $\pm$ sd)}"
        r" & \multicolumn{2}{c}{\makecell{Permutation\\$p$}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
        r"Pool & Edits & Re-rolls & Edits & Re-rolls & Edits & Re-rolls & Mean & Variance \\",
        r"\midrule",
    ]
    for pool in result["pools"]:
        rc = pool["reasoning"]["reasoning_changed"]
        fl = pool["per_question"]["flip_fraction_vs_base"]
        ea, ra = pool["accuracy"]["edits_summary"], pool["accuracy"]["rerolls_incl_base_summary"]
        tm, tv = pool["tests"]["mean_difference"], pool["tests"]["log_variance_ratio"]
        lines.append(
            f"{record_label(pool)} & {fmt(rc['edits']['mean'], 1, 100)} & "
            f"{fmt(rc['rerolls']['mean'], 1, 100)} & "
            f"{fmt(fl['edits']['mean'], 1, 100)} & {fmt(fl['rerolls']['mean'], 1, 100)} & "
            f"${fmt(ea['mean'], 3)} \\pm {fmt(ea['sd_population'], 3)}$ & "
            f"${fmt(ra['mean'], 3)} \\pm {fmt(ra['sd_population'], 3)}$ & "
            f"{fmt_p(tm['p_two_sided'])} & {fmt_p(tv['p_two_sided'])} \\\\")
    pooled = result["pooled"]["all"]["accuracy"]
    lines += [
        r"\midrule",
        f"Pooled & & & & & & & {fmt_p(pooled['mean_difference']['p_two_sided'])} & "
        f"{fmt_p(pooled['log_variance_ratio']['p_two_sided'])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _spread(values: Sequence[float], x0: float, step: float = 0.13) -> list[float]:
    """Deterministic horizontal offsets that fan out tied values around x0."""
    seen: dict[float, int] = {}
    xs = []
    for v in values:
        k = seen.get(v, 0)
        seen[v] = k + 1
        xs.append(x0 + step * ((k + 1) // 2) * (1 if k % 2 else -1))
    return xs


def draw_figure(result: dict, path: Path) -> None:  # pragma: no cover - needs matplotlib
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 7})
    edit_color, roll_color, ink = "#2a78d6", "#eb6834", "0.15"
    pools = result["pools"]
    span = 0.13  # the same y-span (accuracy units) in every panel, so spreads compare by eye
    fig, axes = plt.subplots(1, len(pools), figsize=(5.5, 1.85))
    for ax, pool in zip(axes, pools, strict=True):
        acc = pool["accuracy"]
        edits = sorted(acc["edits"].values())
        rolls = [r["accuracy"] for r in acc["rerolls"]]
        base = acc["base"]
        center = statistics.fmean([*edits, *rolls, base])
        ax.axhline(base, color="0.6", lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.scatter(_spread(edits, 0.0), edits, s=11, color=edit_color, edgecolors="white",
                   linewidths=0.4, zorder=3)
        roll_x = _spread([*rolls, base], 1.0)  # the base fans out from re-rolls it ties with
        ax.scatter(roll_x[:-1], rolls, s=11, color=roll_color, edgecolors="white",
                   linewidths=0.4, zorder=3)
        ax.scatter(roll_x[-1:], [base], s=24, marker="D", facecolors="white",
                   edgecolors=roll_color, linewidths=1.1, zorder=4)
        for x, values in ((0.0, edits), (1.0, [*rolls, base])):
            m = statistics.fmean(values)
            ax.plot([x - 0.32, x + 0.32], [m, m], color=ink, lw=1.0, zorder=2)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(center - span / 2, center + span / 2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["edits", "re-rolls"])
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.tick_params(length=2, pad=1.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(record_label(pool), fontsize=7, pad=2)
        p_mean = fmt_p(pool["tests"]["mean_difference"]["p_two_sided"]).replace("$<$", "<")
        p_var = fmt_p(pool["tests"]["log_variance_ratio"]["p_two_sided"]).replace("$<$", "<")
        ax.text(0.5, 0.02, f"$p_{{\\mathrm{{mean}}}}$ {p_mean}   $p_{{\\mathrm{{var}}}}$ {p_var}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=6, color="0.25")
    axes[0].set_ylabel("held-out accuracy")
    handles = [
        Line2D([], [], ls="", marker="o", ms=3.5, color=edit_color, label="single-token edit"),
        Line2D([], [], ls="", marker="o", ms=3.5, color=roll_color, label="numerical re-roll"),
        Line2D([], [], ls="", marker="D", ms=4, mfc="white", mec=roll_color, label="base read"),
        Line2D([], [], color=ink, lw=1.0, label="mean"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print("wrote", path)


def print_summary(result: dict) -> None:
    for pool in result["pools"]:
        ea, ra = pool["accuracy"]["edits_summary"], pool["accuracy"]["rerolls_incl_base_summary"]
        tm, tv = pool["tests"]["mean_difference"], pool["tests"]["log_variance_ratio"]
        fl = pool["per_question"]["flip_fraction_vs_base"]
        print(f"{record_label(pool)}: edits {ea['mean']:.4f} +- {ea['sd_population']:.4f} | "
              f"re-rolls {ra['mean']:.4f} +- {ra['sd_population']:.4f} | "
              f"p mean {tm['p_two_sided']:.4f} p var {tv['p_two_sided']:.4f} "
              f"(log ratio {tv['observed']:+.3f}) | flips {fl['edits']['mean']:.4f} vs "
              f"{fl['rerolls']['mean']:.4f} p {fl['test_mean_difference']['p_two_sided']:.4f} | "
              f"checks {all(pool['checks'].values())}")
    for subset, tests in result["pooled"].items():
        acc = tests["accuracy"]
        print(f"pooled {subset}: p mean {acc['mean_difference']['p_two_sided']:.4f} "
              f"p var {acc['log_variance_ratio']['p_two_sided']:.4f} "
              f"(sum z {acc['log_variance_ratio']['observed_sum_z']:+.2f}) | "
              f"flips p {tests['flip_fraction']['p_two_sided']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="reroll3 directory")
    parser.add_argument("--json", type=Path, help="write the analysis here")
    parser.add_argument("--from-json", type=Path, help="skip the analysis; render from this file")
    parser.add_argument("--latex", type=Path, help="write the booktabs tabular here")
    parser.add_argument("--figure", type=Path, help="write the strip plot here (needs matplotlib)")
    parser.add_argument("--max-enumerate", type=int, default=1_000_000)
    parser.add_argument("--n-random", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.from_json:
        result = _read(args.from_json)
    else:
        result = analyze(args.dir, args.max_enumerate, args.n_random, args.seed)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8",
                                 newline="\n")
            print("wrote", args.json)
    print_summary(result)
    if args.latex:
        write_latex(result, args.latex)
        print("wrote", args.latex)
    if args.figure:
        draw_figure(result, args.figure)


if __name__ == "__main__":
    main()
