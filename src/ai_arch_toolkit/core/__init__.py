"""Core primitives — zero opinion, maximum flexibility."""

from __future__ import annotations

import logging

from ai_arch_toolkit.core._batch import BatchRequest, BatchResult
from ai_arch_toolkit.core._concurrency import inference_limit
from ai_arch_toolkit.core._content import (
    CachePart,
    Content,
    ContentPart,
    DocumentPart,
    ImagePart,
    assistant,
    cache,
    document,
    image,
    system,
    tool_result,
    user,
)
from ai_arch_toolkit.core._deprecation import deprecated
from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._metering import (
    AdmissionController,
    AdmissionDecision,
    AdmissionDenied,
    Cost,
    CostKind,
    EventStatus,
    MeterSnapshot,
    Money,
    NotMeteredOperationError,
    OperationRequest,
    Pricer,
    Reservation,
    ResourceLimits,
    RunConfig,
    UsageEvent,
    UsageSink,
)
from ai_arch_toolkit.core._metering._scope import MeterScope
from ai_arch_toolkit.core._middleware import Middleware, Request
from ai_arch_toolkit.core._moderation import ModerationError, ModerationResult, Moderator
from ai_arch_toolkit.core._policy import OnExhausted, OnLowConfidence, OnTimeout, Policy
from ai_arch_toolkit.core._pricing import pricing
from ai_arch_toolkit.core._rate_limit import RateLimitMiddleware
from ai_arch_toolkit.core._redaction import (
    RedactionMode,
    RedactionPolicy,
    Redactor,
    TraceMode,
    redact,
    redact_text,
)
from ai_arch_toolkit.core._response import (
    Attempt,
    Citation,
    OutputSchema,
    Response,
    RichStreamResponse,
    StreamEvent,
    SyncRichStreamResponse,
    ThinkingBlock,
    ToolCall,
    Usage,
)
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.core._server_tools import ServerTool, code_execution, web_search
from ai_arch_toolkit.core._state import MergeConflictError, MergeStrategy, State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step, StepFn
from ai_arch_toolkit.core._step_engine import execute_step
from ai_arch_toolkit.core._sync import configure_sync_timeouts
from ai_arch_toolkit.core._telemetry import TracingMiddleware
from ai_arch_toolkit.core._tokens import (
    chars_to_tokens,
    count_tokens_local,
    count_tokens_local_batch,
    tokens_to_chars,
)
from ai_arch_toolkit.core._tools import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalHandler,
    ApprovalRequest,
    DangerousToolGate,
    DryRunGate,
    GovernanceOutcome,
    RiskLevel,
    RunState,
    ToolDefinition,
    ToolError,
    ToolGroup,
    ToolResult,
    ToolRuntimePolicy,
    ToolSchema,
    async_execute_tool,
    execute_tool,
    infer_schema,
    prepare_tools,
    tool,
    tool_schema,
)
from ai_arch_toolkit.core._trace import PolicyDecision, StepTrace, Trace
from ai_arch_toolkit.core.graph import (
    Edge as GraphEdge,
)
from ai_arch_toolkit.core.graph import (
    Graph,
    NodeID,
    NodeType,
)
from ai_arch_toolkit.core.graph import (
    GraphAlgorithms as GraphAlgorithmsProto,
)
from ai_arch_toolkit.core.graph import (
    GraphBackend as GraphBackendProto,
)
from ai_arch_toolkit.core.graph import (
    Node as GraphNode,
)

# Keep package logger configured on import.
_pkg_logger = logging.getLogger("ai_arch_toolkit")
if not any(isinstance(h, logging.NullHandler) for h in _pkg_logger.handlers):
    _pkg_logger.addHandler(logging.NullHandler())

__all__ = [
    "LLM",
    "APIError",
    "AdmissionController",
    "AdmissionDecision",
    "AdmissionDenied",
    "ApprovalDecision",
    "ApprovalGate",
    "ApprovalHandler",
    "ApprovalRequest",
    "Attempt",
    "BatchRequest",
    "BatchResult",
    "CachePart",
    "Citation",
    "Content",
    "ContentPart",
    "Cost",
    "CostKind",
    "DangerousToolGate",
    "DocumentPart",
    "DryRunGate",
    "EventStatus",
    "GovernanceOutcome",
    "Graph",
    "GraphAlgorithmsProto",
    "GraphBackendProto",
    "GraphEdge",
    "GraphNode",
    "ImagePart",
    "MergeConflictError",
    "MergeStrategy",
    "MeterScope",
    "MeterSnapshot",
    "Middleware",
    "ModerationError",
    "ModerationResult",
    "Moderator",
    "Money",
    "NodeID",
    "NodeType",
    "NotMeteredOperationError",
    "OnExhausted",
    "OnLowConfidence",
    "OnTimeout",
    "OperationRequest",
    "OutputSchema",
    "Policy",
    "PolicyDecision",
    "Pricer",
    "RateLimitError",
    "RateLimitMiddleware",
    "RedactionMode",
    "RedactionPolicy",
    "Redactor",
    "Request",
    "Reservation",
    "ResourceLimits",
    "Response",
    "Result",
    "RetryConfig",
    "RichStreamResponse",
    "RiskLevel",
    "RunConfig",
    "RunState",
    "ServerTool",
    "State",
    "StateSnapshot",
    "Step",
    "StepFn",
    "StepTrace",
    "StreamEvent",
    "SyncRichStreamResponse",
    "ThinkingBlock",
    "ToolCall",
    "ToolDefinition",
    "ToolError",
    "ToolGroup",
    "ToolResult",
    "ToolRuntimePolicy",
    "ToolSchema",
    "Trace",
    "TraceMode",
    "TracingMiddleware",
    "Usage",
    "UsageEvent",
    "UsageSink",
    "assistant",
    "async_execute_tool",
    "cache",
    "chars_to_tokens",
    "code_execution",
    "configure_sync_timeouts",
    "count_tokens_local",
    "count_tokens_local_batch",
    "deprecated",
    "document",
    "execute_step",
    "execute_tool",
    "image",
    "infer_schema",
    "inference_limit",
    "prepare_tools",
    "pricing",
    "redact",
    "redact_text",
    "system",
    "tokens_to_chars",
    "tool",
    "tool_result",
    "tool_schema",
    "user",
    "web_search",
]
