"""Sync check: C (repo) vs D (portable) must be identical for all patched/new files.

Guards against the stage-5 failure mode where a patch silently did not take effect because
only one of the two engine copies was updated.
Exit code 0 = all same, 1 = drift found.
"""
import hashlib
import os
import sys

A = r"C:\Users\X303\Documents\FreetokenP\freetoken"
B = r"D:\FreeTokenPortable\engine\venv\Lib\site-packages\freetoken"

PKG_FILES = [
    r"kvcache\mha_pool.py",
    r"kvcache\qsa_pool.py",
    r"kvcache\linear_state_pool.py",
    r"scheduler\cache.py",
    r"scheduler\prefill.py",
    r"scheduler\__init__.py",
    r"cli.py",
    r"kv_swap\__init__.py",
    r"kv_swap\manager.py",
    r"kv_swap\ssd.py",
    r"kv_swap\runtime.py",
]
TOP_FILES = [
    r"start-freetoken-flash-next.ps1",
    r"kv-rebuild-after-ready.ps1",
    r"stop_server_isolated.ps1",
    r"PATCHES.md",
    r"docs\KV_SWAP.md",
]

A_TOP = r"C:\Users\X303\Documents\FreetokenP"
B_TOP = r"D:\FreeTokenPortable"


def h(p):
    if not os.path.exists(p):
        return None
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


bad = 0
print("=== package files (C vs D) ===")
for f in PKG_FILES:
    ha, hb = h(os.path.join(A, f)), h(os.path.join(B, f))
    tag = "SAME" if (ha and ha == hb) else ("MISS" if not ha or not hb else "DIFF")
    if tag != "SAME":
        bad += 1
    print(f"  {tag}  {f}")

print("=== top-level files (C vs D) ===")
for f in TOP_FILES:
    ha, hb = h(os.path.join(A_TOP, f)), h(os.path.join(B_TOP, f))
    tag = "SAME" if (ha and ha == hb) else ("MISS" if not ha or not hb else "DIFF")
    if tag != "SAME":
        bad += 1
    print(f"  {tag}  {f}")

print(f"=== RESULT: {'OK' if bad == 0 else str(bad) + ' DRIFT(S)'} ===")
sys.exit(0 if bad == 0 else 1)
