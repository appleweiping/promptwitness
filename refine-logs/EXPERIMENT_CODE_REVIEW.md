# Structured-gradient pilot: provisional code review

The local `experiment-bridge` workflow calls for a secondary Codex code review
before deploying the comparison. This same-family review is provisional and is
not independent scientific validation.

The reviewer identified and the implementation repaired these blocking paths:

- Search previously materialized holdout labels; it now opens only independently
  prepared fit and validation shards and checks their prepared byte digests.
- The token proposal prefix previously omitted the question; the frozen input
  now precedes the editable blocks, with each row's actual token position used.
- Token arm B previously selected only by four-row accuracy; it now evaluates
  fresh-response answer CE and accepts a lower-loss token candidate.
- EOS at the generation cap was misclassified as truncation; raw output IDs and
  terminal EOS are now carried explicitly.
- Finite differences could normalize a zero gradient; the smoke gate now checks
  a nonzero coordinate, finite values, direction, and magnitude.
- Interrupted or failed operations lacked durable attempted-work accounting;
  the cost journal reserves attempts before work and records partial unknown
  consumption. Generation, forward, and backward stages have separate counters.
- A budget stop could discard completed better candidates; the best completed
  candidate now survives, and incomplete work is marked.
- A token proposal accounting check mixed per-round and cumulative totals; it
  now compares the current round's delta.

The comparison budget remains a research condition: run a real one-round
calibration, freeze a positive optimization cap and overrun tolerance, then
report actual consumed wall time and model token work. Runs with cap zero are
calibration only. No Pilot outcome has been inferred from this review.
