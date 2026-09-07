"""PromptWitness public API."""

from .adapters import (
    AdapterError,
    AdapterFormat,
    AdapterResult,
    adapt_prompt,
    load_adapted_prompt,
    prompt_to_dict,
    render_prompt_json,
)
from .diff import DiffOptions, MessageAlignment, compare_prompts
from .execution import ExecutionReport, ExecutionRow, ScenarioExecutor
from .matrix import (
    MatrixArtifact,
    MatrixDiff,
    RenderedScenario,
    Scenario,
    compare_matrices,
    render_matrix,
    save_matrix,
)
from .models import (
    Change,
    ChangeKind,
    DiffReport,
    Finding,
    FindingCode,
    Message,
    PromptDocument,
    Severity,
    ToolSpec,
    ValidationReport,
)
from .parser import PromptFormatError, load_prompt, parse_prompt
from .policies import PolicyBundle, PolicyFormatError, load_policy, parse_policy
from .providers import (
    OpenAICompatibleProvider,
    OpenAICompatibleStreamingProvider,
    ProviderTrace,
    ReplayProvider,
    TraceEvent,
    TraceRecorder,
    load_traces,
)
from .registry import PromptRegistry, PromptVersion, RenderedPrompt
from .tools import ToolBatch, ToolDispatcher, ToolOutcome
from .validation import ValidationPolicy, validate_prompt

__all__ = [
    "AdapterError",
    "AdapterFormat",
    "AdapterResult",
    "Change",
    "ChangeKind",
    "DiffOptions",
    "DiffReport",
    "ExecutionReport",
    "ExecutionRow",
    "Finding",
    "FindingCode",
    "MatrixArtifact",
    "MatrixDiff",
    "Message",
    "MessageAlignment",
    "OpenAICompatibleProvider",
    "OpenAICompatibleStreamingProvider",
    "PolicyBundle",
    "PolicyFormatError",
    "PromptDocument",
    "PromptFormatError",
    "PromptRegistry",
    "PromptVersion",
    "ProviderTrace",
    "RenderedPrompt",
    "RenderedScenario",
    "ReplayProvider",
    "Scenario",
    "ScenarioExecutor",
    "Severity",
    "ToolBatch",
    "ToolDispatcher",
    "ToolOutcome",
    "ToolSpec",
    "TraceEvent",
    "TraceRecorder",
    "ValidationPolicy",
    "ValidationReport",
    "adapt_prompt",
    "compare_matrices",
    "compare_prompts",
    "load_adapted_prompt",
    "load_policy",
    "load_prompt",
    "load_traces",
    "parse_policy",
    "parse_prompt",
    "prompt_to_dict",
    "render_matrix",
    "render_prompt_json",
    "save_matrix",
    "validate_prompt",
]

__version__ = "0.2.0"
