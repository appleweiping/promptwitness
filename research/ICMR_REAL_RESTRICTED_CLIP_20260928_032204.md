# ICMR composed-image retrieval: real CLIP in the restricted search role

Recorded 2026-09-28 03:22:04 UTC. This is an **authored-input engineering
qualification**, not an official CIRR/FashionIQ evaluation, M1/M2 admission,
Pilot result, method comparison, or ICMR paper result. The original five-seed
and six-configuration confirmation plan remains unrun. The ARIS default four
rounds, historical cost ledger, no-timer decision and four frozen repositories
are unchanged.

## What actually ran

On native WSL2 Ubuntu (`Linux 6.18.33.2-microsoft-standard-WSL2`, Landlock
ABI 7), a fresh private clone of this branch ran `reproduce.check_retrieval_rank_real_clip`
with `CUDA_VISIBLE_DEVICES=""`. The checkpoint was the first-party OpenAI CLIP
ViT-L/14 file, 932,768,134 bytes, SHA-256
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.
It was downloaded and hash-verified into an owned private directory, then
hard-linked into a private input-only leaf. Neither weight nor raw images are
committed or redistributed. The first-party [OpenAI CLIP source](https://github.com/openai/CLIP)
is the previously pinned `fbea8bf4` tree, staged privately into this isolated
environment; the download used that source's `ViT-L/14` loader URL. Pinned
runtime observed: Python 3.12, torch
2.7.1+cpu, torchvision 0.22.1+cpu, NumPy 1.26.4, SciPy 1.15.3, Pillow 11.3.0.
The official checkpoint download/verification had separately recorded
155.037709 seconds; it is not included in the checker wall clock.
The qualified native copies match the pending checkout sources by SHA-256:
checker `c5270ac39bf817e3b09a9179d6e7baf3164a941538000b607b2c974e675c0382`,
policy `83f9a1cf2b12490d28768bfc0ae24f81cc4889390c833547c92b76121cb6aaf1`.

The script creates only three local pattern images (RGB red, RGBA blue,
grayscale gray) and two authored English modifications. It launches a persistent
Landlock `retrieval_search_ranker` reading `search/inputs`, builds one real CLIP
gallery index, ranks both queries by query ID, then passes the complete rankings
to a separate gold-reading restricted scorer. The checker did not read official
benchmark annotations/images or submit to a test server. Fusion weight 0.5 is
mechanically fixed for this witness, **not scientifically frozen**. The raw
rankings were `[red, blue, gray]` and `[blue, red, gray]`; CIRR-style scoring
removes each reference first, so both authored targets had hit@1. These tiny
constructed hits provide no evidence of retrieval quality or method benefit.

The successful private `qualification.json` has SHA-256
`3ab6c5839dc2655e88ccca80d0593ceb924312c1b301b7163514f6817b6a2939`.
Its status is `PASS_AUTHORED_REAL_CLIP_RESTRICTED_SEARCH_AND_SCORER`; ranker
and scorer were distinct PIDs. Inner receipts: 8 attempts, 8 completed,
0 failed/unresolved, 5 known encoder forward calls (3 images, 2 texts),
3 operations with unmeasured forward count (load and two ranks). Outer receipts:
3 attempts/completed, 0 failed/unresolved. Checker wall clock: 58.455116 s;
inner operation wall **sum is nonadditive** (44.067700 s), not a physical total.
Whole-process allocated CPU cost and all-role forecast remain unmeasured, not zero.
No new LLM calls, API spend or GPU allocation were made in this qualification.

## Failure history and narrow repair

Two distinct fresh private outputs were retained, never overwritten:

1. `qualification-primary` stopped before ranker readiness because PyTorch
   tried opening `/dev/urandom` under Landlock. Retained JSON SHA-256:
   `92b38421d03501eed7024da7e0ba0fbc244ebbd68e3fc2d50843233417db3bab`.
2. After a ranker-only, read-only grant for that file, `qualification-secondary`
   stopped while converting CLIP weights because the CPU library could not
   read `/proc/cpuinfo`. Retained JSON SHA-256:
   `32b36982098ab7dd71726baab8b892f1b8ad7317ff1bfd5911ae46214ddcc76b`.

The final policy grants only those two files read access to the retrieval
ranker, not `/dev` or `/proc` directories, and does not grant them to the scorer.
An actual Linux worker test verifies ranker can read both but cannot read
`search/gold`; scorer can read its authorized gold but neither runtime file.
This filesystem boundary does not imply network isolation, provenance,
resistance to adaptive score probing, or secrecy from the trusted controller.

A fresh same-family/provisional code review found that a later scorer failure
would lose completed rankings from the failure JSON. The checker now records
each successful ranking immediately and retains scorer observations before
receipt assertions. A scorer-failure regression checks `FAILED_RETAINED`,
both rankings and the failed outer scoring attempt. The one focused follow-up
review closed that blocker; it was not an independent cross-family or
scientific review.

## Verification and next admission gates

Before this record: native Linux focused suite `45 passed, 1 skipped` in 73.13 s,
including true Landlock workers; Windows new-driver test `3 passed`; Ruff check
and formatting pass; source mypy passes 61 files. The full Windows suite was
still running when this timestamped report was written; its result and the
new exact-SHA CI must be recorded separately, not backdated here.

Still required: lawful and versioned CIRR/FashionIQ image/annotation sources;
train-side grouping and sealed final; official evaluator full-gallery/tie parity;
captioner plus both original generation models; actual fit/selection/final
roles and end-to-end physical cost forecast; native online optimizer/control
path; preregistered Pilot and confirmatory comparisons against strong baselines.
Current M1/Pilot/scientific admission is `NOT_ADMITTED`.
