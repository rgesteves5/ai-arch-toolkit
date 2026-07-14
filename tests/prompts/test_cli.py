"""Prompt CLI tests without model or network calls."""

from __future__ import annotations

import json
from pathlib import Path

from ai_arch_toolkit._cli import main


def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "cli.prompt.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "name": "cli-test",
                "variables": {
                    "name": {"type": "string", "required": True},
                    "count": {"type": "integer", "default": 1},
                },
                "sections": [
                    {
                        "name": "request",
                        "template": {"content": "Hello ${name} ${count}"},
                    }
                ],
            }
        )
    )
    return path


def test_prompt_validate(tmp_path: Path, capsys) -> None:
    status = main(["prompt", "validate", str(manifest(tmp_path))])

    captured = capsys.readouterr()
    assert status == 0
    assert "valid prompt 'cli-test': 1 sections, 2 variables" in captured.out


def test_prompt_inspect_is_structured_and_secret_free(tmp_path: Path, capsys) -> None:
    status = main(["prompt", "inspect", str(manifest(tmp_path))])

    data = json.loads(capsys.readouterr().out)
    assert status == 0
    assert data["name"] == "cli-test"
    assert data["sections"][0]["engine"] == "string-template"
    assert data["variables"][0]["name"] == "name"


def test_prompt_render_with_inline_and_file_variables(tmp_path: Path, capsys) -> None:
    variables = tmp_path / "variables.json"
    variables.write_text(json.dumps({"name": "Ada", "count": 2}))

    status = main(
        [
            "prompt",
            "render",
            str(manifest(tmp_path)),
            "--vars",
            str(variables),
            "--var",
            "count=3",
            "--layout",
            "text",
        ]
    )

    assert status == 0
    assert capsys.readouterr().out == "Hello Ada 3\n"


def test_prompt_render_error_returns_two(tmp_path: Path, capsys) -> None:
    status = main(["prompt", "render", str(manifest(tmp_path))])

    captured = capsys.readouterr()
    assert status == 2
    assert "missing required prompt variables" in captured.err


def test_invalid_assignment_and_variables_file(tmp_path: Path, capsys) -> None:
    path = manifest(tmp_path)
    status = main(["prompt", "render", str(path), "--var", "invalid"])
    assert status == 2
    assert "NAME=VALUE" in capsys.readouterr().err

    variables = tmp_path / "variables.json"
    variables.write_text("[]")
    status = main(["prompt", "render", str(path), "--vars", str(variables)])
    assert status == 2
    assert "must contain an object" in capsys.readouterr().err


def test_cli_loads_knowledge_files_for_manifest(tmp_path: Path, capsys) -> None:
    prompt = tmp_path / "knowledge.prompt.json"
    prompt.write_text(
        json.dumps(
            {
                "version": 1,
                "sections": [{"name": "knowledge", "knowledge": "story.rules"}],
            }
        )
    )
    rules = tmp_path / "rules.txt"
    rules.write_text("Keep continuity.")
    status = main(
        [
            "prompt",
            "render",
            str(prompt),
            "--knowledge",
            f"story.rules={rules}",
        ]
    )
    assert status == 0
    assert capsys.readouterr().out == "Keep continuity.\n"


def test_cli_loads_knowledge_directory(tmp_path: Path, capsys) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "rules.txt").write_text("RULES")
    prompt = tmp_path / "knowledge.prompt.json"
    prompt.write_text(
        json.dumps(
            {
                "version": 1,
                "sections": [{"name": "knowledge", "knowledge": "rules"}],
            }
        )
    )
    status = main(["prompt", "validate", str(prompt), "--knowledge-dir", str(knowledge)])
    assert status == 0
    assert "valid prompt" in capsys.readouterr().out
