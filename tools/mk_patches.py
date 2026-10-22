import difflib
import io
import os

OFFICIAL = r"C:\Users\X303\AppData\Local\FreeToken\venv\Lib\site-packages\freetoken"
REPO = r"C:\Users\X303\Documents\FreetokenP\freetoken"
OUT = r"C:\Users\X303\Documents\FreetokenP\patches"

MODIFIED = [
    "kvcache/mha_pool.py",
    "kvcache/qsa_pool.py",
    "kvcache/linear_state_pool.py",
    "scheduler/cache.py",
    "scheduler/prefill.py",
    "scheduler/__init__.py",
    "cli.py",
]


def read_lines(path):
    """Read text and normalise ALL line endings (\r\n, bare \r, mixed) to \n."""
    t = io.open(path, encoding="utf-8", newline="").read()
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


os.makedirs(OUT, exist_ok=True)
for rel in MODIFIED:
    ap = os.path.join(OFFICIAL, rel.replace("/", os.sep))
    bp = os.path.join(REPO, rel.replace("/", os.sep))
    a = read_lines(ap)
    b = read_lines(bp)
    diff = list(
        difflib.unified_diff(
            a, b, fromfile="a/" + rel, tofile="b/" + rel, n=3, lineterm=""
        )
    )
    name = rel.replace("/", "__") + ".diff"
    with io.open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(diff) + "\n")
    added = sum(1 for l in diff if l[:1] == "+" and not l.startswith("+++"))
    removed = sum(1 for l in diff if l[:1] == "-" and not l.startswith("---"))
    print(f"{rel:40} +{added:<5} -{removed:<5} -> {name}")
print("done")
