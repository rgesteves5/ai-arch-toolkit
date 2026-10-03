"""The offline logic of ``scripts/probe_images.py`` (I01); the live calls run by hand."""

from __future__ import annotations

import base64
import struct
import zlib

from scripts.probe_images import (
    Budget,
    Check,
    Outcome,
    Probe,
    data_url,
    dimensions,
    redact,
    render,
    run_checks,
    sniff,
)


def _png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixels = zlib.compress(b"\x00" + b"\x00\x00\x00" * width)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels)


def test_sniff_reads_the_signature_of_each_format() -> None:
    assert sniff(_png(1, 1)) == "image/png"
    assert sniff(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert sniff(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert sniff(b"GIF89a") == "unknown"


def test_dimensions_come_from_the_png_and_webp_headers() -> None:
    assert dimensions(_png(1536, 864)) == (1536, 864)
    canvas = (1599).to_bytes(3, "little") + (1599).to_bytes(3, "little")
    extended = b"RIFF\x00\x00\x00\x00WEBPVP8X" + b"\x00" * 8 + canvas
    assert dimensions(extended) == (1600, 1600)
    lossy = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 10 + struct.pack("<HH", 1152, 2016)
    assert dimensions(lossy) == (1152, 2016)
    bits = (1919) | (1279 << 14)
    lossless = (
        b"RIFF\x00\x00\x00\x00WEBPVP8L" + b"\x00" * 5 + bits.to_bytes(4, "little") + b"\x00" * 5
    )
    assert dimensions(lossless) == (1920, 1280)
    assert dimensions(b"\xff\xd8\xff\xe0rest") is None


def test_data_url_round_trips_the_bytes() -> None:
    url = data_url(b"abc", "image/png")
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"abc"


def test_redact_hides_keys_and_organizations() -> None:
    text = redact("bad key sk-proj-abc123 for org-XYZ and AIzaSyAbc_def and xai-QWE123")
    assert "abc123" not in text and "XYZ" not in text
    assert "SyAbc" not in text and "QWE123" not in text


def test_a_check_over_the_cap_is_skipped_and_costs_nothing(tmp_path) -> None:
    ran: list[str] = []

    def run(probe: Probe) -> str:
        ran.append("x")
        return "ok"

    checks = [
        Check(number="1", provider="p", question="cheap", worst_usd=0.6, run=run),
        Check(number="2", provider="p", question="dear", worst_usd=0.6, run=run),
    ]
    budget = Budget(cap=1.0)
    outcomes = run_checks(checks, Probe(out_dir=tmp_path), budget)
    assert [o.skipped for o in outcomes] == [False, True]
    assert ran == ["x"]
    assert budget.spent == 0.6


def test_a_failing_check_is_recorded_as_its_answer(tmp_path) -> None:
    def fails(probe: Probe) -> str:
        raise RuntimeError("400: unsupported size sk-secret1")

    def needs_earlier(probe: Probe) -> str:
        return probe.kept["first"]

    checks = [
        Check(number="1", provider="p", question="q", worst_usd=0.1, run=fails),
        Check(number="2", provider="p", question="q", worst_usd=0.1, run=needs_earlier),
    ]
    outcomes = run_checks(checks, Probe(out_dir=tmp_path), Budget(cap=1.0))
    assert outcomes[0].outcome.startswith("error RuntimeError: 400: unsupported size")
    assert "secret1" not in outcomes[0].outcome
    assert outcomes[1].outcome == "not run: needs 'first' from an earlier check"


def test_save_keeps_the_first_image_of_each_provider_for_edits(tmp_path) -> None:
    probe = Probe(out_dir=tmp_path)
    first, second = _png(2, 2), _png(3, 3)
    probe.save("openai-gpt-image-2", first)
    probe.save("openai-edit", second)
    assert probe.images["openai"] == (first, "image/png")
    assert (tmp_path / "openai-edit.png").read_bytes() == second


def test_render_lists_every_outcome() -> None:
    outcomes = [Outcome(number="1a", provider="openai", question="usage?", outcome="yes")]
    text = render(outcomes, Budget(cap=2.0, spent=0.03), "20261003T000000Z")
    assert "$0.03 of $2.00" in text
    assert "## 1a · openai · usage?" in text
