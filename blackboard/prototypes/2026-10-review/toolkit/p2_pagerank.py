import asyncio
import networkx as nx
from ai_arch_toolkit.core.graph import Graph, Node
from ai_arch_toolkit.core.graph._networkx import NetworkXBackend

async def main():
    g = Graph(NetworkXBackend())
    for n in "ABC":
        await g.add(Node(id=n))
    await g.connect("A", "B", "knows")
    await g.connect("A", "B", "likes")
    await g.connect("A", "C", "knows")
    await g.connect("B", "A", "knows")
    await g.connect("C", "A", "knows")
    pr = await g.pagerank()
    print({k: round(v, 4) for k, v in pr.items()}, "sum", round(sum(pr.values()), 4))
    try:  # networkx's own pagerank needs numpy, which the dev extra does not install
        print("networkx:", {k: round(v, 4) for k, v in nx.pagerank(g.backend._graph).items()})
    except ImportError as exc:
        print("networkx comparison skipped:", exc)
asyncio.run(main())
