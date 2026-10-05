"""The memory graph has a public backend, of the type its protocol takes (G-24)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ai_arch_toolkit.toolkit.memory import Node
from ai_arch_toolkit.toolkit.memory.graph import GraphStore, MemoryBackend, NetworkXBackend

USE = """\
from ai_arch_toolkit.toolkit.memory import NetworkXBackend as FromMemory, Node
from ai_arch_toolkit.toolkit.memory.graph import GraphStore, MemoryBackend, NetworkXBackend
from ai_arch_toolkit.core.graph import NetworkXBackend as CoreBackend, GraphBackend

store = GraphStore(NetworkXBackend())
other = GraphStore(FromMemory())
backend: MemoryBackend = NetworkXBackend()
graph: GraphBackend = CoreBackend()


async def read(node_id: str) -> Node | None:
    return await NetworkXBackend().get_node(node_id)
"""


def test_the_backend_is_in_the_public_packages() -> None:
    import ai_arch_toolkit.core.graph as core_graph
    import ai_arch_toolkit.toolkit.memory as memory
    import ai_arch_toolkit.toolkit.memory.graph as memory_graph

    assert "NetworkXBackend" in memory_graph.__all__
    assert "NetworkXBackend" in memory.__all__
    assert "NetworkXBackend" in core_graph.__all__
    assert memory.NetworkXBackend is memory_graph.NetworkXBackend


def test_it_satisfies_the_memory_protocol_at_runtime() -> None:
    assert isinstance(NetworkXBackend(), MemoryBackend)


async def test_a_stored_node_comes_back_as_a_memory_node() -> None:
    store = GraphStore(NetworkXBackend())
    node = await store.add(Node(type="fact", content={"text": "Lisbon is the capital"}))

    found = await store.get(node.id)

    assert isinstance(found, Node)
    assert found.content == {"text": "Lisbon is the capital"}


def test_the_type_checker_accepts_it(tmp_path: Path) -> None:
    pyright = Path(sys.executable).parent / "pyright"
    if not pyright.exists():
        pytest.skip("pyright is not installed (the dev extra)")
    use = tmp_path / "use_memory.py"
    use.write_text(USE)

    checked = subprocess.run(
        [str(pyright), str(use)], capture_output=True, text=True, timeout=120, check=False
    )

    assert checked.returncode == 0, checked.stdout


def test_without_networkx_asking_for_it_says_which_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_arch_toolkit.toolkit.memory.graph as memory_graph

    for name in [m for m in sys.modules if m.endswith("graph._networkx")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "networkx", None)

    with pytest.raises(ImportError, match=r"ai-arch-toolkit\[graph\]"):
        memory_graph.__getattr__("NetworkXBackend")
