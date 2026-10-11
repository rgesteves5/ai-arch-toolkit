import asyncio
from ai_arch_toolkit.toolkit.memory import GraphStore, Node, SimilarityView
from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend

async def embed(text):
    # "cat" texts point one way, everything else slightly off
    return [1.0, 0.0] if "cat" in text else [0.9, 0.1]

async def main():
    store = GraphStore(NetworkXBackend(), embed=embed)
    for i in range(20):
        await store.add(Node(type="fact", content={"text": f"cat fact {i}"}))
    await store.add(Node(type="rule", content={"text": "always feed the dog"}))
    rules = SimilarityView(store, node_type="rule")
    print("rule nodes in store:", await store.count(type="rule"))
    print("SimilarityView(rule).find('cat', k=3):", await rules.find("cat", k=3))
asyncio.run(main())
