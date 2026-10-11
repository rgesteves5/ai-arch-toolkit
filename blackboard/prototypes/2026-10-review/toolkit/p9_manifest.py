import json, pathlib, sys
from ai_arch_toolkit.toolkit.agents import load_agent_manifest, AgentManifestError
import tempfile
base = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()).resolve()
root = base / "mf" / "root"
root.mkdir(parents=True, exist_ok=True)
(base / "mf" / "outside" / "x:").mkdir(parents=True, exist_ok=True)  # b) needs a directory named "x:"
(base / "mf" / "outside" / "x:" / "secret.txt").write_text("SECRET OUTSIDE ROOT\n")
def write(name, data):
    p = root / name; p.write_text(json.dumps(data)); return p
# a) null means "not set" elsewhere, but not here
for label, body in [
    ("model.temperature: null", {"version": 1, "model": {"model": "m", "temperature": None}}),
    ("strategy.knobs: null", {"version": 1, "strategy": {"name": "react", "knobs": None}}),
    ("model.max_tokens: null", {"version": 1, "model": {"model": "m", "max_tokens": None}}),
]:
    p = write("a.agent.json", body)
    try:
        load_agent_manifest(p); print("a)", label, "-> accepted")
    except AgentManifestError as e:
        print("a)", label, "->", e)
# b) system_file with '://' skips the allowed-roots check and the fingerprint
outside = (base / "mf" / "outside").resolve()
evil = str(outside) + "/x://secret.txt"
p = write("b.agent.json", {"version": 1, "strategy": {"name": "plan_execute", "phases": {"planner": {"system_file": evil}}}})
try:
    m = load_agent_manifest(p, allowed_roots=[root])
    print("b) loaded; referenced fingerprints:", dict(m.referenced_fingerprints))
    print("b) planner_system =", repr(m.reasoning_spec().knobs.get("planner_system")))
except Exception as e:
    print("b) refused:", type(e).__name__, e)
