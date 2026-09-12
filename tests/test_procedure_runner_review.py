"""Independent regression probes for model identity at the invocation boundary."""

import pytest
from test_procedure_plan import ANSWER, small_plan

from promptwitness import (
    OpenAICompatibleProvider,
    OpenAITaskProvider,
    TaskRunConflict,
    TaskRunStore,
)


def test_openai_task_invocation_uses_a_private_configuration_snapshot(monkeypatch, tmp_path):
    """A concurrent caller edit must not change the transport of an existing request."""
    original = OpenAICompatibleProvider(
        "http://127.0.0.1:8000/v1/chat/completions",
        model="pinned-model",
        headers={"X-Experiment": "pinned-header"},
    )
    observed = []

    def local_complete(current, messages, *, tools=(), generation=None):
        # Controlled stand-in for a concurrent owner changing the public provider.
        # There is no HTTP call: observe the exact configured object to be used.
        original.model = "different-model"
        original.endpoint = "http://127.0.0.1:9000/v1/chat/completions"
        original.headers["X-Experiment"] = "different-header"
        observed.append(
            {
                "model": current.model,
                "endpoint": current.endpoint,
                "headers": dict(current.headers),
                "generation": dict(generation),
            }
        )
        return ANSWER

    monkeypatch.setattr(OpenAICompatibleProvider, "complete", local_complete)
    plan = small_plan(generation={"max_tokens": 79, "temperature": 0})
    with TaskRunStore(tmp_path / "snapshot.sqlite") as store:
        store.bind(plan, OpenAITaskProvider(original).identity)
        request, revision = store.reserve(plan.payload["items"][0]["id"], expected_revision=0)
        # Direct invocation isolates transport snapshot semantics from run's
        # separate post-callback identity check, which should reject drift.
        response = OpenAITaskProvider(original)(request)
        assert response == ANSWER
        assert observed == [
            {
                "model": "pinned-model",
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "headers": {"X-Experiment": "pinned-header"},
                "generation": {"max_tokens": 79, "temperature": 0},
            }
        ]
        assert revision == 1


@pytest.mark.parametrize(
    "replacement",
    [
        {"provider": "local", "model": "changed"},
        None,
        {"provider": "local", "model": object()},
        "unreadable",
    ],
)
def test_post_callback_identity_drift_stays_uncertain_without_score_or_replay(
    tmp_path, replacement
):
    class MutableProvider:
        def __init__(self):
            self.current_identity = {"provider": "local", "model": "pinned"}
            self.calls = 0

        @property
        def identity(self):
            if self.current_identity == "unreadable":
                raise RuntimeError("private identity lookup failed")
            return self.current_identity

        def __call__(self, request):
            self.calls += 1
            self.current_identity = replacement
            return ANSWER

    provider = MutableProvider()
    plan = small_plan()
    database = tmp_path / "drift.sqlite"
    with TaskRunStore(database) as store:
        with pytest.raises(TaskRunConflict):
            store.run(plan, provider)
        report = store.snapshot()
        row = report["results"][0]
        assert row["status"] == "running"
        assert row["revision"] == 1
        assert row["attempts"] == 1
        assert row["result"] is None
        assert report["coverage"]["succeeded"] == 0
        assert report["coverage"]["scored"] == 0
        assert provider.calls == 1

    fresh_provider = MutableProvider()
    with TaskRunStore(database) as reopened:
        assert reopened.run(plan, fresh_provider) == report
    assert fresh_provider.calls == 0
