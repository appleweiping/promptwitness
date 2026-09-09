# Scripted multi-interview demonstration

This is an original, entirely synthetic engineering fixture. Mira and Noel are
invented participants in an invented lantern-walkway project. The responses are
authored Python data: no model, credentials, network call, paid service, or real
participant is involved. Successful execution demonstrates lifecycle and source
binding, not interview quality, factual accuracy, or research equivalence.

These are development-branch examples, not a claim that the APIs are available
on `main` or in a released package. From a checkout of
`feat/whole-repository-alignment`, install the checkout and run:

```shell
python -m pip install -e .
python examples/interview_demo.py
```

The default creates a fresh temporary directory, runs against real SQLite,
closes every connection, removes only that temporary directory, and prints an
aggregate JSON report. To keep the database and report, choose a **new** directory
whose parent already exists:

```shell
python examples/interview_demo.py --output-dir ./my-new-interview-demo
```

The command refuses an existing directory. It never resumes or overwrites a
previous run. `interviews.sqlite` is a private transcript artifact containing
the synthetic questions, answers, source quotes, original completed provider
envelopes, requests and memories. `summary.json` contains counts, event kinds,
coverage denominators and hashes, not raw answers or envelopes. A failure may
leave a partial new directory for inspection; rerun into a different new path.

## Hand-authored workflow

1. Mira receives a route-preparation question. Her answer contains an emoji,
   CRLF, and literal `{{template}}`. The database is closed at revision 4 and
   reopened before analysis.
2. Analysis cites that exact answer and proposes an accessibility topic. At
   revision 6, the scheduler waits for explicit review; only one question has
   been published. The example explicitly accepts the proposal at revision 7.
3. The second required topic, lamp checks, is completed first. At revision 12,
   required coverage is 2/2 but accepted-emergent coverage is 0/1. The scheduler
   continues into the accepted quiet-lane question instead of stopping early.
4. Mira finishes at revision 18 with three source-linked memories, three
   answers, required coverage 2/2 and emergent coverage 1/1. Six scripted stage
   calls occurred; they are not model calls.
5. Noel completes a separate one-topic interview at revision 7, producing an
   actual committed memory for the isolation check. Attempting to import both
   participants' heads into Mira's return interview is rejected without creating
   that interview.
6. After another close/reopen, Mira's return interview imports only her three
   committed memories. Its first question cites all three, using the deliberately
   shared fixture vocabulary. New required coverage is still 0/2. The example
   explicitly stops this return interview at revision 4; it does not falsely
   report its agenda as complete.

The complete fixture has 29 events across three interviews, nine scripted
provider calls (five questioner, four analyst), zero model calls, and two database
reopens. The summary publishes independently checkable checkpoint revisions and
export hashes. Hashes depend on the exact development templates, schemas,
request rendering and fixture text; they are not signatures or proof of truth.

The provider emits native closed response JSON with an authored `stop` envelope.
The ordinary runner and reducer parse and retain it; the example neither imports
test helpers nor manufactures a transport completion for an actual model call.
The schema's `model_proposed` summary label is preserved, but this fixture's
summary text is explicitly scripted and not model-generated.

Tests in `tests/test_interview_demo.py` assert the hand-counted trace, exact source
text and spans, review boundary, separate coverage denominators, same-participant
recall, rejected mixed provenance, exclusive output creation, and the absence of
network/model calls. This does not evaluate open-ended conversation, semantic
retrieval quality, memory correction, or a real interviewer's capabilities.
