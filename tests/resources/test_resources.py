"""Resource resolver, codec, policy, and provenance tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_arch_toolkit.toolkit.resources import (
    Resource,
    ResourceDecodeError,
    ResourceLoadError,
    ResourcePolicy,
    ResourcePolicyError,
    ResourceRef,
    ResourceResolver,
    ResourceTooLargeError,
    load_resource,
    load_resources,
)


def test_resource_from_text_preserves_raw_text_and_fingerprint() -> None:
    resource = Resource.from_text("Olá 🌍", uri="memory://greeting")

    assert resource.raw == "Olá 🌍".encode()
    assert resource.text == "Olá 🌍"
    assert resource.data == "Olá 🌍"
    assert resource.fingerprint == "sha256:" + hashlib.sha256(resource.raw).hexdigest()
    assert resource.provenance is not None
    assert resource.provenance.loader == "memory"


def test_resource_from_bytes_preserves_binary_memory_content() -> None:
    resource = Resource.from_bytes(b"\x00\x01", uri="memory://fixture")
    assert resource.data == b"\x00\x01"
    assert resource.text is None
    assert resource.provenance is not None
    assert resource.provenance.loader == "memory"
    with pytest.raises(TypeError, match="raw must be bytes"):
        Resource.from_bytes("bad")  # type: ignore[arg-type]


def test_resource_metadata_is_copied_and_read_only() -> None:
    metadata = {"kind": "prompt"}
    resource = Resource.from_text("text", metadata=metadata)
    metadata["kind"] = "changed"

    assert resource.metadata["kind"] == "prompt"
    with pytest.raises(TypeError):
        resource.metadata["kind"] = "blocked"  # type: ignore[index]


def test_resource_ref_validation() -> None:
    with pytest.raises(ValueError, match="uri is required"):
        ResourceRef(uri="")
    with pytest.raises(TypeError, match="media_type"):
        ResourceRef(uri="x", media_type=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="encoding"):
        ResourceRef(uri="x", encoding="")


def test_resource_value_validation() -> None:
    ref = ResourceRef(uri="memory://value")
    with pytest.raises(TypeError, match="raw must be bytes"):
        Resource(ref=ref, raw="x", media_type="text/plain", data="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="media_type is required"):
        Resource(ref=ref, raw=b"x", media_type="", data="x")
    with pytest.raises(TypeError, match="text must be"):
        Resource(ref=ref, raw=b"x", media_type="text/plain", data="x", text=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="metadata must be"):
        Resource(ref=ref, raw=b"x", media_type="text/plain", data="x", metadata=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="fingerprint must be"):
        Resource(ref=ref, raw=b"x", media_type="text/plain", data="x", fingerprint=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="provenance must be"):
        Resource(ref=ref, raw=b"x", media_type="text/plain", data="x", provenance="x")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="from_text text"):
        Resource.from_text(1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("filename", "content", "media_type", "data"),
    [
        ("value.txt", "hello", "text/plain", "hello"),
        ("value.md", "# Hello", "text/markdown", "# Hello"),
        ("value.json", '{"a": 1}', "application/json", {"a": 1}),
        ("value.toml", "a = 1\n", "application/toml", {"a": 1}),
        ("value.yaml", "a: 1\n", "application/yaml", {"a": 1}),
    ],
)
def test_load_resource_decodes_known_formats(
    tmp_path: Path,
    filename: str,
    content: str,
    media_type: str,
    data: object,
) -> None:
    path = tmp_path / filename
    path.write_text(content)

    resource = load_resource(path)

    assert resource.media_type == media_type
    assert resource.text == content
    assert resource.data == data
    assert resource.provenance is not None
    assert resource.provenance.resolved_uri == str(path.resolve())
    assert resource.provenance.byte_length == len(content.encode())


def test_media_type_override_decodes_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "content.unknown"
    path.write_text('{"valid": true}')

    resource = load_resource(path, media_type="application/json")

    assert resource.data == {"valid": True}


def test_unknown_extension_uses_binary_codec(tmp_path: Path) -> None:
    path = tmp_path / "content.unknown"
    path.write_bytes(b"\x00\x01")

    resource = load_resource(path)

    assert resource.media_type == "application/octet-stream"
    assert resource.text is None
    assert resource.data == b"\x00\x01"


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("bad.json", "{", "invalid JSON"),
        ("bad.toml", "[bad", "invalid TOML"),
        ("bad.yaml", "key: [", "invalid YAML"),
    ],
)
def test_invalid_structured_resource_has_clear_error(
    tmp_path: Path, filename: str, content: str, message: str
) -> None:
    path = tmp_path / filename
    path.write_text(content)

    with pytest.raises(ResourceDecodeError, match=message):
        load_resource(path)


def test_invalid_text_encoding_has_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff")

    with pytest.raises(ResourceDecodeError, match="could not decode"):
        load_resource(path)


def test_missing_file_is_wrapped(tmp_path: Path) -> None:
    with pytest.raises(ResourceLoadError, match="could not read resource"):
        load_resource(tmp_path / "missing.txt")


def test_resource_policy_restricts_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("no")
    policy = ResourcePolicy(allowed_roots=(allowed,))

    with pytest.raises(ResourcePolicyError, match="outside allowed roots"):
        load_resource(outside, policy=policy)


def test_resource_policy_rejects_large_files(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("1234")

    with pytest.raises(ResourceTooLargeError, match="maximum is 3"):
        load_resource(path, policy=ResourcePolicy(max_bytes=3))


def test_resource_policy_validation() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ResourcePolicy(max_bytes=0)
    with pytest.raises(ValueError, match="positive integer"):
        ResourcePolicy(max_bytes=True)
    with pytest.raises(TypeError, match="allow_remote"):
        ResourcePolicy(allow_remote=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="allow_symlinks"):
        ResourcePolicy(allow_symlinks=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="allow_absolute_paths"):
        ResourcePolicy(allow_absolute_paths=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="allowed_media_types"):
        ResourcePolicy(allowed_media_types=frozenset({""}))


def test_resource_policy_can_restrict_absolute_paths_and_media_types(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("text")
    with pytest.raises(ResourcePolicyError, match="absolute resource paths"):
        load_resource(path, policy=ResourcePolicy(allow_absolute_paths=False))
    with pytest.raises(ResourcePolicyError, match="media type"):
        load_resource(
            path,
            policy=ResourcePolicy(allowed_media_types=frozenset({"application/json"})),
        )


def test_symlink_can_be_disabled(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("content")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are not available")

    with pytest.raises(ResourcePolicyError, match="symbolic links"):
        load_resource(link, policy=ResourcePolicy(allow_symlinks=False))


def test_remote_resources_are_disabled_before_loader_lookup() -> None:
    with pytest.raises(ResourceLoadError, match="remote resources are disabled"):
        load_resource("https://example.com/prompt.txt")


def test_unknown_scheme_lists_registered_loaders() -> None:
    with pytest.raises(ResourceLoadError, match="no resource loader"):
        load_resource(ResourceRef(uri="custom://prompt"))


def test_package_resource_loader_reads_packaged_schema() -> None:
    resource = load_resource(
        ResourceRef(
            uri=(
                "package://ai_arch_toolkit/toolkit/prompts/schemas/prompt-manifest-v1.schema.json"
            )
        )
    )
    assert resource.data["properties"]["version"] == {"const": 1}
    assert resource.provenance is not None
    assert resource.provenance.loader == "package"


@pytest.mark.parametrize(
    "uri",
    [
        "package://ai_arch_toolkit",
        "package:///schema.json",
        "package://ai_arch_toolkit/../pyproject.toml",
        "package://ai_arch_toolkit/%2e%2e/pyproject.toml",
        "package://missing_package/value.json",
        "package://ai_arch_toolkit/missing.json",
    ],
)
def test_invalid_package_resources_are_wrapped(uri: str) -> None:
    with pytest.raises(ResourceLoadError):
        load_resource(ResourceRef(uri=uri))


def test_load_resource_rejects_conflicting_resolver_and_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="either resolver or policy"):
        load_resource(
            tmp_path / "x.txt",
            resolver=ResourceResolver(),
            policy=ResourcePolicy(),
        )


def test_resource_ref_rejects_separate_overrides(tmp_path: Path) -> None:
    ref = ResourceRef.from_path(tmp_path / "x.txt")

    with pytest.raises(ValueError, match="overrides cannot be used"):
        load_resource(ref, media_type="text/plain")
    with pytest.raises(ValueError, match="overrides cannot be used"):
        load_resource(ref, encoding="latin-1")


def test_custom_codec_and_extension(tmp_path: Path) -> None:
    class UpperCodec:
        name = "upper"

        def decode(self, raw, ref):
            from ai_arch_toolkit.toolkit.resources._codecs import DecodedResource

            text = raw.decode(ref.encoding).upper()
            return DecodedResource(data=text, text=text)

    path = tmp_path / "value.upper"
    path.write_text("hello")
    resolver = ResourceResolver()
    resolver.register_codec("text/x-upper", UpperCodec(), extensions=("upper",))

    resource = resolver.resolve(path)

    assert resource.media_type == "text/x-upper"
    assert resource.data == "HELLO"
    assert resource.provenance.codec == "upper"


def test_custom_serializer_is_isolated_per_resolver() -> None:
    class UpperSerializer:
        name = "upper"

        def serialize(self, value):
            return str(value).upper()

    resolver = ResourceResolver()
    resolver.register_serializer("upper", UpperSerializer())
    assert resolver.serializers.resolve("upper").serialize("hello") == "HELLO"
    with pytest.raises(ValueError, match="unknown resource serializer"):
        ResourceResolver().serializers.resolve("upper")


def test_codec_registration_validation() -> None:
    resolver = ResourceResolver()
    with pytest.raises(ValueError, match="media type"):
        resolver.register_codec("", object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="extensions"):
        resolver.register_codec("text/x-test", object(), extensions=("",))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="scheme"):
        resolver.register_loader(1, object())  # type: ignore[arg-type]


def test_custom_loader() -> None:
    class CustomLoader:
        name = "custom"

        def load(self, ref, policy):
            raw = b'{"loaded": true}'
            policy.check_size(len(raw), uri=ref.uri)
            return raw, ref.uri

    resolver = ResourceResolver()
    resolver.register_loader("custom", CustomLoader())

    resource = resolver.resolve(ResourceRef(uri="custom://value", media_type="application/json"))

    assert resource.data == {"loaded": True}
    assert resource.provenance.loader == "custom"


def test_load_directory_is_sorted_by_relative_path(tmp_path: Path) -> None:
    (tmp_path / "z").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "z" / "same.txt").write_text("z")
    (tmp_path / "a" / "same.txt").write_text("a")

    resources = load_resources(tmp_path, recursive=True)

    assert [Path(resource.ref.uri).relative_to(tmp_path).as_posix() for resource in resources] == [
        "a/same.txt",
        "z/same.txt",
    ]


def test_load_directory_filters_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.json").write_text(json.dumps({"b": 1}))

    resources = load_resources(tmp_path, extensions={".json"})

    assert len(resources) == 1
    assert resources[0].data == {"b": 1}


def test_load_resources_accepts_existing_resolver_and_rejects_policy_too(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A")
    resolver = ResourceResolver(policy=ResourcePolicy(allowed_roots=(tmp_path,)))
    assert load_resources(tmp_path, resolver=resolver)[0].text == "A"
    with pytest.raises(ValueError, match="either resolver or policy"):
        load_resources(tmp_path, resolver=resolver, policy=ResourcePolicy())


def test_explicit_unregistered_media_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("x")
    with pytest.raises(ResourceDecodeError, match="no resource codec"):
        load_resource(path, media_type="text/x-missing")


def test_yaml_dependency_error_is_actionable(tmp_path: Path) -> None:
    path = tmp_path / "data.yaml"
    path.write_text("key: value")

    with patch.dict("sys.modules", {"yaml": None}), pytest.raises(ImportError, match=r"\[yaml\]"):
        load_resource(path)
