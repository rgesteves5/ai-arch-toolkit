import pathlib
import sys, asyncio
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "tests"))  # the repo root, 4 levels up
from fake_provider import fake_llm
from ai_arch_toolkit.toolkit.memory import MemoryMiddleware, SearchResult, Node

async def find(query, k=3):
    return [SearchResult(node=Node(content={"text": "user likes cats"}), score=1.0)]
async def record(d):
    return None

async def main():
    for mw in ([], [MemoryMiddleware(find=find, record=record)]):
        llm, prov = fake_llm(images=True, middleware=mw)
        try:
            await llm.generate_image("a red bicycle")
        except Exception as e:
            print("raised", type(e).__name__, e)
        last = prov.last
        print("middleware:", [type(m).__name__ for m in mw], "-> image request kept:", last.image is not None, "| system:", repr(last.system))
asyncio.run(main())
