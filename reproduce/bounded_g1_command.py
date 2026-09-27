"""Run one owned CPU command with an immutable conservative core-hour grant."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from contextlib import suppress
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 1 <= args.timeout <= 1800 or not args.command:
        raise ValueError("bounded command, 1..4 workers and 1..1800 seconds required")
    import psutil

    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.open("a+", encoding="utf-8") as ledger:
        ledger.seek(0)
        past = [json.loads(line) for line in ledger if line.strip()]
        # Reservations persist before spawning; a crashed owner cannot grant itself
        # a fresh 80 hours. Charge the timeout*worker upper envelope conservatively.
        used = sum(row["reserved_cpu_core_seconds"] for row in past)
        reservation = args.timeout * args.workers
        if used + reservation > 80 * 3600:
            raise ValueError("offline G1 conservative CPU grant exhausted")
        event = {
            "command": args.command,
            "workers": args.workers,
            "reserved_cpu_core_seconds": reservation,
            "started_unix": time.time(),
        }
        ledger.write(json.dumps(event, sort_keys=True) + "\n")
        ledger.flush()
        os.fsync(ledger.fileno())
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = str(args.workers)
    started = time.monotonic()
    process = subprocess.Popen(args.command, env=env)
    owned = psutil.Process(process.pid)
    cpu_seen: dict[tuple[int, float], float] = {}
    timed_out = False
    while process.poll() is None:
        try:
            descendants = [owned, *owned.children(recursive=True)]
            for member in descendants:
                try:
                    times = member.cpu_times()
                    cpu_seen[member.pid, member.create_time()] = times.user + times.system
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except psutil.NoSuchProcess:
            descendants = []
        if time.monotonic() - started > args.timeout:
            # Only this launched process and its confirmed descendants, never
            # name-based kills or shared-host job enumeration.
            for member in reversed(descendants):
                with suppress(psutil.NoSuchProcess):
                    member.kill()
            timed_out = True
            break
        time.sleep(0.1)
    exit_code = process.wait()
    print(
        json.dumps(
            {
                "command_exit_code": exit_code,
                "timed_out": timed_out,
                "elapsed_seconds": time.monotonic() - started,
                "sampled_process_tree_cpu_seconds_lower_observation": sum(cpu_seen.values()),
                "conservatively_charged_cpu_core_seconds": reservation,
                "cumulative_conservatively_reserved_cpu_core_hours": (used + reservation) / 3600,
                "short_lived_child_cpu_sampling_may_be_incomplete": True,
            }
        ),
        flush=True,
    )
    raise SystemExit(124 if timed_out else exit_code)


if __name__ == "__main__":
    main()
