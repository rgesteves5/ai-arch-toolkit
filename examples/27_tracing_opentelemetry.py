"""27 — Tracing Middleware (OpenTelemetry-style, OpenAI).

Demonstrates TracingMiddleware with:
  - OpenTelemetry SDK tracer when available
  - fallback console tracer when SDK packages are not installed
"""

from __future__ import annotations

from ai_arch_toolkit import Client, TracingMiddleware


class _ConsoleSpan:
    def set_attribute(self, key, value):
        print(f"[trace] {key}={value}")

    def record_exception(self, exception):
        print(f"[trace] exception={type(exception).__name__}: {exception}")


class _ConsoleSpanContextManager:
    def __init__(self, name: str) -> None:
        self._name = name
        self._span = _ConsoleSpan()

    def __enter__(self):
        print(f"[trace] start span={self._name}")
        return self._span

    def __exit__(self, exc_type, exc, tb):
        print(f"[trace] end span={self._name}")


class _ConsoleTracer:
    def start_as_current_span(self, name: str):
        return _ConsoleSpanContextManager(name)


def _build_tracer():
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        print("Using OpenTelemetry SDK tracer with ConsoleSpanExporter.")
        return trace.get_tracer("examples.tracing")
    except Exception:
        print("OpenTelemetry SDK not installed; using local console tracer fallback.")
        return _ConsoleTracer()


middleware = TracingMiddleware(tracer=_build_tracer())
client = Client("openai", model="gpt-5-nano", middleware=[middleware])

print("=== traced chat ===")
response = client.chat("In two sentences, explain CAP theorem.")
print(response.text)

print("\n=== traced stream_events ===")
for event in client.stream_events("Name three queueing systems and one use-case each."):
    if event.type == "text":
        print(event.text, end="", flush=True)
print("\n[stream complete]")
