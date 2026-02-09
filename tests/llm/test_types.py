"""Tests for _types.py dataclasses."""

from ai_arch_toolkit.llm._types import (
    AudioPart,
    DocumentPart,
    ImagePart,
    JsonSchema,
    Message,
    Response,
    ServerTool,
    StreamEvent,
    TextPart,
    ThinkingBlock,
    ThinkingConfig,
    Tool,
    ToolCall,
    ToolResult,
    Usage,
)


def test_message_creation() -> None:
    m = Message(role="user", content="hello")
    assert m.role == "user"
    assert m.content == "hello"


def test_message_defaults() -> None:
    m = Message(role="user")
    assert m.content == ""
    assert m.tool_calls == ()


def test_message_with_tool_calls() -> None:
    tc = ToolCall(id="1", name="f", arguments={"a": 1})
    m = Message(role="assistant", content="thinking", tool_calls=(tc,))
    assert len(m.tool_calls) == 1
    assert m.tool_calls[0].name == "f"


def test_message_with_tuple_content() -> None:
    parts = (TextPart("hello"), ImagePart(url="https://example.com/img.png"))
    m = Message(role="user", content=parts)
    assert isinstance(m.content, tuple)
    assert len(m.content) == 2
    assert isinstance(m.content[0], TextPart)
    assert isinstance(m.content[1], ImagePart)


def test_message_string_content_backward_compat() -> None:
    m = Message(role="user", content="hello")
    assert isinstance(m.content, str)
    assert m.content == "hello"


def test_tool_result_creation() -> None:
    tr = ToolResult(tool_call_id="call_1", name="get_weather", content='{"temp": 20}')
    assert tr.tool_call_id == "call_1"
    assert tr.name == "get_weather"
    assert tr.content == '{"temp": 20}'


def test_tool_creation() -> None:
    t = Tool(name="f", description="desc", parameters={"type": "object"})
    assert t.name == "f"
    assert t.parameters == {"type": "object"}


def test_tool_call_creation() -> None:
    tc = ToolCall(id="1", name="f", arguments={"a": 1})
    assert tc.id == "1"
    assert tc.arguments == {"a": 1}


def test_usage_defaults() -> None:
    u = Usage()
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.total_tokens == 0
    assert u.cache_creation_tokens == 0
    assert u.cache_read_tokens == 0


def test_usage_values() -> None:
    u = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
    assert u.total_tokens == 30


def test_usage_cache_fields() -> None:
    u = Usage(input_tokens=100, cache_creation_tokens=50, cache_read_tokens=30)
    assert u.cache_creation_tokens == 50
    assert u.cache_read_tokens == 30


def test_response_defaults() -> None:
    r = Response()
    assert r.text == ""
    assert r.tool_calls == ()
    assert r.stop_reason == ""
    assert r.thinking == ""
    assert r.thinking_blocks == ()
    assert r.raw == {}
    assert isinstance(r.usage, Usage)


def test_response_tool_calls_is_tuple() -> None:
    tc = ToolCall(id="1", name="f", arguments={})
    r = Response(tool_calls=(tc,))
    assert isinstance(r.tool_calls, tuple)
    assert len(r.tool_calls) == 1


def test_response_with_values() -> None:
    tc = ToolCall(id="1", name="f", arguments={})
    r = Response(
        text="hi",
        tool_calls=(tc,),
        usage=Usage(10, 20, 30),
        stop_reason="stop",
        raw={"id": "x"},
    )
    assert r.text == "hi"
    assert len(r.tool_calls) == 1
    assert r.usage.total_tokens == 30


def test_response_with_thinking() -> None:
    tb = ThinkingBlock(text="Let me think...")
    r = Response(
        text="The answer is 42.",
        thinking="Let me think...",
        thinking_blocks=(tb,),
    )
    assert r.thinking == "Let me think..."
    assert len(r.thinking_blocks) == 1
    assert r.thinking_blocks[0].text == "Let me think..."


def test_response_to_message() -> None:
    tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})
    r = Response(text="Let me check.", tool_calls=(tc,))
    msg = r.to_message()
    assert msg.role == "assistant"
    assert msg.content == "Let me check."
    assert msg.tool_calls == (tc,)


def test_response_to_message_no_tool_calls() -> None:
    r = Response(text="Hello!")
    msg = r.to_message()
    assert msg.role == "assistant"
    assert msg.content == "Hello!"
    assert msg.tool_calls == ()


def test_json_schema_creation() -> None:
    schema = JsonSchema(
        name="person",
        schema={"type": "object", "properties": {"name": {"type": "string"}}},
    )
    assert schema.name == "person"
    assert schema.strict is True


def test_json_schema_non_strict() -> None:
    schema = JsonSchema(
        name="data",
        schema={"type": "object"},
        strict=False,
    )
    assert schema.strict is False


# --- Multimodal content part tests ---


def test_text_part() -> None:
    p = TextPart("hello world")
    assert p.text == "hello world"


def test_image_part_url() -> None:
    p = ImagePart(url="https://example.com/img.png", media_type="image/png")
    assert p.url == "https://example.com/img.png"
    assert p.detail == "auto"


def test_image_part_base64() -> None:
    p = ImagePart(data="abc123==", media_type="image/jpeg", detail="high")
    assert p.data == "abc123=="
    assert p.detail == "high"


def test_audio_part() -> None:
    p = AudioPart(data="audiodata==", media_type="audio/wav")
    assert p.data == "audiodata=="
    assert p.media_type == "audio/wav"
    assert p.transcript == ""


def test_document_part() -> None:
    p = DocumentPart(data="pdfdata==")
    assert p.media_type == "application/pdf"
    assert p.uri == ""


def test_document_part_uri() -> None:
    p = DocumentPart(uri="gs://bucket/file.pdf")
    assert p.uri == "gs://bucket/file.pdf"


# --- Thinking config tests ---


def test_thinking_config_defaults() -> None:
    tc = ThinkingConfig()
    assert tc.effort == "medium"
    assert tc.budget_tokens is None


def test_thinking_config_custom() -> None:
    tc = ThinkingConfig(effort="high", budget_tokens=16384)
    assert tc.effort == "high"
    assert tc.budget_tokens == 16384


def test_thinking_block() -> None:
    tb = ThinkingBlock(text="reasoning text here")
    assert tb.text == "reasoning text here"


# --- Stream event tests ---


def test_stream_event_text() -> None:
    e = StreamEvent(type="text", text="hello")
    assert e.type == "text"
    assert e.text == "hello"
    assert e.tool_call is None
    assert e.usage is None


def test_stream_event_tool_call() -> None:
    tc = ToolCall(id="1", name="f", arguments={})
    e = StreamEvent(type="tool_call", tool_call=tc)
    assert e.type == "tool_call"
    assert e.tool_call is tc


def test_stream_event_thinking() -> None:
    e = StreamEvent(type="thinking", thinking="hmm")
    assert e.thinking == "hmm"


def test_stream_event_usage() -> None:
    u = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
    e = StreamEvent(type="usage", usage=u)
    assert e.usage is u


def test_stream_event_done() -> None:
    e = StreamEvent(type="done")
    assert e.type == "done"


# --- Server tool tests ---


def test_server_tool() -> None:
    st = ServerTool(type="web_search")
    assert st.type == "web_search"
    assert st.config == {}


def test_server_tool_with_config() -> None:
    st = ServerTool(type="web_search", config={"allowed_domains": ["example.com"]})
    assert st.config["allowed_domains"] == ["example.com"]
