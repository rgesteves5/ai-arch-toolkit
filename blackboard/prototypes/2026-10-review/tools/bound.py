import os, sys, pathlib
from ai_arch_toolkit import ToolGroup, ApprovalDecision
from ai_arch_toolkit.core import ToolCall
from ai_arch_toolkit.toolkit.tools.dangerous import FilesystemPolicy, filesystem_tools, PathScopeGate
import tempfile
base = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()).resolve()
(base / "root").mkdir(exist_ok=True); (base / "outside").mkdir(exist_ok=True)
(base / "outside" / "secret.txt").write_text("SECRET")
(base / "root" / "ok.txt").write_text("fine")
link = base / "root" / "link"
if not link.exists(): link.symlink_to(base / "outside")
policy = FilesystemPolicy(read_roots=(base / "root",), write_roots=(base / "root",), cwd=base / "root")
tools = filesystem_tools(policy)
names = [getattr(t, "__name__", "?") for t in tools]
print(names)
group = ToolGroup(*tools, approval_handler=lambda r: ApprovalDecision.approve())
def call(name, **args):
    r = group.execute(ToolCall(id="1", name=name, input=args))
    print(name, args, "->", r.ok, (r.value if r.ok else r.error.type + ": " + r.error.message)[:110])
call("read_file", path=str(base / "outside" / "secret.txt"))
call("read_file", path="link/secret.txt")
call("read_file", path="ok.txt")
call("list_directory", path=".", pattern="link/*")
call("search_files", directory=".", pattern="SECRET")
call("write_file", path="link/x.txt", content="x")
call("move_path", source="ok.txt", destination="../outside/ok.txt")
