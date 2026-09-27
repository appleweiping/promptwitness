"""AUTHORED cost fixtures, NEVER actual research history or inference."""

import hashlib
import json
import sqlite3

import pytest

from promptwitness.incremental.budget import ResourceLedger
from reproduce.online_resources import CEILING, GPU, STAGE, closed_history, continue_ledger


def authored(path):
    return ResourceLedger(
        path,
        historical_usage={
            "calls": 898,
            "input_tokens": 1758607,
            "output_tokens": 190976,
            "gpu_hours": 3.4812718904634763,
        },
        historical_digest="a" * 64,
        global_limit=CEILING,
        stage_limits={"AUTHORED": CEILING},
        gpu_uuid=GPU,
    )


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_continue_closed_actual_shape_not_reset(tmp_path):
    path = tmp_path / "AUTHORED_NEVER_REAL.sqlite"
    ledger = authored(path)
    ledger.reserve_call("one", "AUTHORED", "b" * 64, input_cap=100, output_cap=10)
    ledger.settle_call("one", succeeded=True, input_tokens=40, output_tokens=4)
    ledger.reserve_call("failure", "AUTHORED", "c" * 64, input_cap=200, output_cap=20)
    ledger.settle_call("failure", succeeded=False, input_tokens=None, output_tokens=None)
    ledger.begin_gpu("one", "AUTHORED", GPU, start=100)
    ledger.end_gpu("one", end=3700)
    ledger.close()
    original = path.read_bytes()
    history = closed_history(path, checksum(path))
    assert path.read_bytes() == original
    assert history["usage"] == {
        "calls": 900,
        "input_tokens": 1758847,
        "output_tokens": 191000,
        "gpu_hours": 4.4812718904634763,
    }
    assert history["original_local_calls_with_reserved_or_unknown_token_cost"] == 1
    dest = tmp_path / "AUTHORED_CONTINUED.sqlite"
    continued = continue_ledger(dest, history)
    assert continued.usage()["calls"] == 900
    assert continued.specification["stages"][STAGE] == continued.specification["global_limit"]
    continued.reserve_call("new", STAGE, "d" * 64, input_cap=30, output_cap=5)
    continued.settle_call("new", succeeded=False, input_tokens=21, output_tokens=0)
    assert continued.usage()["calls"] == 901
    assert continued.usage()["input_tokens"] == 1758868
    continued.close()
    continued = continue_ledger(dest, history)
    assert continued.usage()["calls"] == 901
    continued.close()


@pytest.mark.parametrize("open_kind", ["call", "gpu"])
def test_unresolved_history_never_closed_as_zero(tmp_path, open_kind):
    path = tmp_path / "AUTHORED.sqlite"
    ledger = authored(path)
    if open_kind == "call":
        ledger.reserve_call("pending", "AUTHORED", "b" * 64, input_cap=10, output_cap=2)
    else:
        ledger.begin_gpu("open", "AUTHORED", GPU, start=100)
    ledger.close()
    with pytest.raises(ValueError, match="unresolved"):
        closed_history(path, checksum(path))


def test_hash_spec_status_and_history_format_rejected(tmp_path):
    path = tmp_path / "AUTHORED.sqlite"
    ledger = authored(path)
    ledger.reserve_call("one", "AUTHORED", "b" * 64, input_cap=10, output_cap=2)
    ledger.settle_call("one", succeeded=True, input_tokens=1, output_tokens=1)
    ledger.close()
    with pytest.raises(ValueError, match="changed"):
        closed_history(path, "0" * 64)
    with sqlite3.connect(path) as database:
        database.execute("UPDATE model_calls SET status='invented'")
    with pytest.raises(ValueError, match="unknown original"):
        closed_history(path, checksum(path))
    with sqlite3.connect(path) as database:
        spec = json.loads(database.execute("SELECT specification FROM resource_spec").fetchone()[0])
        spec["gpu_uuid"] = "GPU-unassigned"
        database.execute("UPDATE resource_spec SET specification=?", (json.dumps(spec),))
    with pytest.raises(ValueError, match="assigned device"):
        closed_history(path, checksum(path))
    with pytest.raises(ValueError, match="continuous"):
        continue_ledger(tmp_path / "other.sqlite", {"format": "AUTHORED"})
