# Exact-SHA CI for the real restricted CLIP witness

Recorded 2026-09-28 03:35:38 UTC. Commit
`bf627a6b1be27b0ae06219cf0e0be1a5f30fc876` was pushed to
`research/promptwitness-delta-v1`; the remote branch was read back at that
same SHA and the working tree was clean. Its [GitHub Actions run
36373923799](https://github.com/appleweiping/promptwitness/actions/runs/36373923799)
ended `completed/success`, **15/15 jobs success**. Ubuntu Python 3.10 ran
`2057 passed, 2 skipped`, coverage 94.04%; Ubuntu Python 3.14 ran
`2057 passed, 2 skipped, 127 warnings`, coverage 94.05%. Quality, build,
wheel smoke, macOS and Windows matrix jobs also ended success. The warnings
remain visible in CI and are not represented as zero.

The same checked-out source had local Windows full `pytest --no-cov -q`
exit 0, current targeted access/driver tests `37 passed, 3 skipped`,
Ruff check/format pass, mypy on 61 package source files pass, configured
source Bandit pass, and helper mypy pass with explicit package bases. On an
owned native Linux WSL clone, the related role/driver suite was
`45 passed, 1 skipped`; a separate true CPU CLIP qualification used identical
checker/policy SHA-256 to the committed source and retained its outputs.

CI itself does **not** download the 932 MB checkpoint or run the real CLIP
model: it validates authored/fake role regressions and the Linux Landlock
boundary. The separate WSL witness establishes that the first-party CLIP
model can run with three authored images and two authored query texts inside
that restricted ranker, not that official CIRR/FashionIQ data, evaluator
parity, C1–C3 effects, complete CPU allocation, M1 or Pilot are accepted.
Those gates remain `NOT_ADMITTED`; the failed first two WSL attempts and all
historical research/cost/access records remain retained. No new timer,
ARIS-round change, paid API or work on the other four repositories occurred.
