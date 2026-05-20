#!/usr/bin/env python3
"""Generate a one-page app summary PDF without external dependencies."""

from __future__ import annotations

import textwrap
from pathlib import Path

PAGE_W = 612
PAGE_H = 792
MARGIN_X = 54
TOP_Y = 756
BOTTOM_Y = 46


def esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Layout:
    def __init__(self) -> None:
        self.y = TOP_Y
        self.ops: list[str] = []

    def _line(self, text: str, *, x: int, font: str, size: int, leading: int) -> None:
        if self.y < BOTTOM_Y:
            raise RuntimeError("Content overflowed past one page")
        self.ops.append(f"BT /{font} {size} Tf 1 0 0 1 {x} {self.y:.2f} Tm ({esc(text)}) Tj ET")
        self.y -= leading

    def heading(self, text: str) -> None:
        self._line(text, x=MARGIN_X, font="F2", size=13, leading=16)

    def title(self, text: str) -> None:
        self._line(text, x=MARGIN_X, font="F2", size=20, leading=24)

    def subtitle(self, text: str) -> None:
        self._line(text, x=MARGIN_X, font="F1", size=10, leading=14)

    def para(self, text: str, *, size: int = 10) -> None:
        max_chars = 92 if size == 10 else 84
        for line in textwrap.wrap(text, width=max_chars):
            self._line(line, x=MARGIN_X, font="F1", size=size, leading=13)

    def bullet(self, text: str) -> None:
        wrapped = textwrap.wrap(text, width=86)
        if not wrapped:
            return
        self._line(f"- {wrapped[0]}", x=MARGIN_X, font="F1", size=10, leading=13)
        for cont in wrapped[1:]:
            self._line(cont, x=MARGIN_X + 12, font="F1", size=10, leading=13)

    def numbered(self, n: int, text: str) -> None:
        wrapped = textwrap.wrap(text, width=84)
        if not wrapped:
            return
        self._line(f"{n}. {wrapped[0]}", x=MARGIN_X, font="F1", size=10, leading=13)
        for cont in wrapped[1:]:
            self._line(cont, x=MARGIN_X + 16, font="F1", size=10, leading=13)

    def spacer(self, points: int = 6) -> None:
        self.y -= points


def build_pdf(content_stream: bytes) -> bytes:
    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>")
    objects.append(
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
        ).encode("ascii")
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    stream_obj = (
        f"<< /Length {len(content_stream)} >>\nstream\n".encode("ascii")
        + content_stream
        + b"\nendstream"
    )
    objects.append(stream_obj)

    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n")
    pdf.extend(b"%\xe2\xe3\xcf\xd3\n")

    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii"))
    pdf.extend(f"startxref\n{xref_start}\n%%EOF\n".encode("ascii"))
    return bytes(pdf)


def main() -> None:
    out_path = Path("output/pdf/ai-arch-toolkit-summary.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    l = Layout()
    l.title("ai-arch-toolkit: One-page Summary")
    l.subtitle("Repository evidence date: 2026-02-19")
    l.spacer(2)

    l.heading("What it is")
    l.para(
        "ai-arch-toolkit is a Python 3.12+ library with a unified LLM client API across multiple providers."
    )
    l.para(
        "It combines middleware, tool orchestration, and eight agent architectures in one package."
    )
    l.spacer()

    l.heading("Who it is for")
    l.bullet(
        "Primary persona (inferred from docs/examples): Python developers building LLM apps and agent workflows."
    )
    l.bullet("Formal business persona statement: Not found in repo.")
    l.spacer()

    l.heading("What it does")
    l.bullet(
        "Uses one Client/AsyncClient API across OpenAI, Anthropic, Gemini, xAI, Mistral, and Groq."
    )
    l.bullet(
        "Supports chat, text streaming, and typed stream events (text, tool_call, thinking, usage, done)."
    )
    l.bullet("Enables structured output with JSON schemas and multimodal content parts.")
    l.bullet("Provides middleware hooks for caching, cost tracking, guardrails, and tracing.")
    l.bullet(
        "Generates tool schemas with @tool and executes validated calls through ToolRegistry."
    )
    l.bullet("Includes eight agent implementations with shared config/result/event interfaces.")
    l.bullet("Supports sync and async batch APIs for job submit, status, and result retrieval.")
    l.spacer()

    l.heading("How it works (repo-based architecture)")
    l.bullet("Application code or an agent invokes Client/AsyncClient methods.")
    l.bullet("Client normalizes messages and runs middleware before hooks on a Request envelope.")
    l.bullet("create_provider() selects a provider adapter and routes to external LLM APIs.")
    l.bullet("Provider responses pass through middleware after hooks back to caller.")
    l.bullet(
        "For tool use: model receives ToolRegistry definitions, emits ToolCall, registry validates/executes, and ToolResult is fed back into the loop."
    )
    l.bullet(
        "Checkpoint persistence and runtime resume APIs exist as stubs: Not implemented in repo."
    )
    l.spacer()

    l.heading("How to run (minimal)")
    l.numbered(1, "Install uv (see docs/uv-guide.md).")
    l.numbered(2, "Clone the repository and cd into it. Exact clone URL: Not found in repo.")
    l.numbered(3, "Install dependencies: uv sync --dev")
    l.numbered(4, "Load API keys when needed: set -a; source .env; set +a")
    l.numbered(5, "Run an example: uv run python examples/01_hello_world.py")

    if l.y < BOTTOM_Y:
        raise RuntimeError("Content overflowed one page")

    content = ("\n".join(l.ops) + "\n").encode("ascii")
    pdf_bytes = build_pdf(content)
    out_path.write_bytes(pdf_bytes)
    print(out_path)


if __name__ == "__main__":
    main()
