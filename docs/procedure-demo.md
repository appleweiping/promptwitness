# Six-family durable workflow demo

This is an original **synthetic, offline** engineering demonstration, not a model
benchmark. Six small hand-authored cases live in `examples/procedures/`; no
LongProc data or reference evaluator is needed. Countdown and the directed route
use the input-only deterministic baselines. The other four responses are explicit
scripted fixtures. Code is retained as inert text, never compiled or executed.

From the development checkout with the matching package installed:

```powershell
python scripts/procedure_workflow_demo.py
```

This creates a temporary directory, executes the workflow, closes every database
connection, prints an aggregate JSON report, and removes only its own temporary
directory. To retain the actual SQLite journal and aggregate report:

```powershell
python scripts/procedure_workflow_demo.py --directory D:/existing-artifacts/new-procedure-demo --output D:/existing-artifacts/new-procedure-demo/summary.json
```

The chosen directory must not already exist; its parent must exist. An existing
database or output is never overwritten. Outputs cannot be an example source,
the demo database, or one of its SQLite sidecars. The default workflow uses no
subprocess, network, credentials or language-model call. The retained database
contains the invented example inputs and answers; the aggregate report contains
only counts, statuses, hashes and public task-family names.

Help output is plain. On Python 3.14 both the parser and its formatter disable
color probing so an already-closed stdout cannot prevent the workflow from
starting. If the final print fails after a requested report was published, the
command returns status 1 and leaves that report intact; the exit code does not
mean that no work was performed.

## What actually happens

The cases adapter loads six typed cases. Each has a one-byte budget (deliberately
skipped) and an 8,192-byte budget (eligible), giving twelve planned items. Each
task has its own `max_tokens`: 128, 160, 192, 224, 256, and 288. The provider checks
the settings delivered on every request. This demonstrates identity-bound
generation configuration, not a measured tokenizer or real decoder budget.

| Checkpoint | Scripted calls | Durable state |
| --- | ---: | --- |
| Commit Countdown reservation, then close | 0 | 1 running, 5 ready, 6 skipped |
| Reopen and run only ready items | 5 | 1 running, 1 failed, 4 succeeded, 6 skipped |
| Explicitly authorize retries | 7 | 6 succeeded, 6 skipped |
| Run again and reopen again | 7 | Unchanged; no additional calls |

The first path response has `finish_reason="length"`, even though its body looks
valid. It is rejected and never stored as a successful result. The interrupted
Countdown reservation is not retried automatically. After explicit retry and a
new reservation, a late response bearing the old revision is rejected. The new
Countdown attempt then completes, and a separate explicit retry completes the
path task.

There are eight reservation attempts but seven provider calls because the first
Countdown reservation is deliberately interrupted before invocation. The final
journal contains seventeen events. Countdown finishes at revision four; path at
revision five; the other four eligible items finish at revision two. All six
small-budget items retain revision zero.

Five supported primary metrics have the hand-expected value 1.0. Pseudo-code
executable correctness remains `unsupported`, with a null primary score. The
report therefore has five scored items out of twelve planned items, not a claim
of complete six-family scoring. Both six-family completion flags stay false:
skipped items remain in the execution denominator, and the code metric is still
unsupported. Different family metrics are not averaged into an invented overall
accuracy.

## Evidence boundaries

The interruption is a deliberate connection close after a committed reservation,
not an operating-system crash or killed-worker stress test. The demo uses a single
Python process and explicitly knows the old worker has stopped before retrying.
It does not prove exactly-once remote side effects. Fixtures cover duplicate TSV
rows, an emoji city, UTF-8 text transport, literal `{{literal}}` text and
an authored CRLF story. The source fixture/script hashes are checked before and
after the demo; the report is not a full runtime-build attestation.

The independent tests assert the expected counts and revisions directly and read
the persisted event table. Re-running the complete demo in two new directories
produces the same aggregate report. Nothing here measures model quality, general
HTML extraction, belief entailment, compiled-code correctness or whole-repository
parity.
