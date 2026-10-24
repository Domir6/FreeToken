"""Invariant suite: static presence checks + built-in unit tests (+ optional live page-accounting).

Exit 0 = all pass. Run after any patch/sync:
    python tools/invariant_suite.py
"""
import io
import os
import re
import subprocess
import sys

sys.path.insert(0, r"C:\Users\X303\Documents\FreetokenP")
PKG = r"C:\Users\X303\Documents\FreetokenP\freetoken"
PY = sys.executable

ok = True


def check(cond, msg, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + msg + (("  <- " + extra) if (extra and not cond) else ""))
    ok = ok and cond


def src(rel):
    return io.open(os.path.join(PKG, rel), encoding="utf-8", newline="").read()


# ---- 1. static presence of the KV-swap API surface ----
check("def gather_kv(" in src(r"kvcache\mha_pool.py"), "mha_pool.gather_kv present")
check("def scatter_kv(" in src(r"kvcache\mha_pool.py"), "mha_pool.scatter_kv present")
check("def gather_index_tiers(" in src(r"kvcache\qsa_pool.py"), "qsa_pool.gather_index_tiers present")
check("def scatter_index_tiers(" in src(r"kvcache\qsa_pool.py"), "qsa_pool.scatter_index_tiers present")
check("def snapshot_slot(" in src(r"kvcache\linear_state_pool.py"), "linear_state_pool.snapshot_slot present")
check("def restore_slot(" in src(r"kvcache\linear_state_pool.py"), "linear_state_pool.restore_slot present")
check("def attempt_host_restore(" in src(r"scheduler\cache.py"), "cache.attempt_host_restore present")
check("def _try_reattach_mamba(" in src(r"scheduler\cache.py"), "cache._try_reattach_mamba present")
check("class KVAllocationError" in src(r"scheduler\cache.py"), "KVAllocationError present")
check("def capture_resident(" in src(r"kv_swap\manager.py"), "manager.capture_resident present")
check("def capture_node_mamba(" in src(r"kv_swap\manager.py"), "manager.capture_node_mamba present")
check("def _on_entry_dropped(" in src(r"kv_swap\manager.py"), "manager._on_entry_dropped present (neg-cache fix)")
check("_notify_drop" in src(r"kv_swap\ssd.py"), "ssd._notify_drop present (neg-cache fix)")

# ---- 2. suppressed-heuristic must be physically gone ----
resid = []
for dp, dn, fn in os.walk(PKG):
    for f in fn:
        if f.endswith(".py"):
            s = io.open(os.path.join(dp, f), encoding="utf-8", newline="").read()
            if re.search(r"suppress_for|is_suppressed|SUPPRESS_ON_COLD|SUPPRESS_MIN_TOKENS|_suppress_until", s):
                resid.append(f)
check(not resid, "no suppress-heuristic residue", str(resid))
ps1 = io.open(r"C:\Users\X303\Documents\FreetokenP\start-freetoken-flash-next.ps1", encoding="utf-8-sig", newline="").read()
check("SUPPRESS" not in ps1, "no SUPPRESS env in ps1")

# ---- 3. minor fixes ----
check("_KV_DIAG_ENABLED" in src(r"scheduler\cache.py"), "_kv_diag uses module-level flag (a)")
mgr = src(r"kv_swap\manager.py")
blk = mgr.split("def capture_node_mamba(")[1][:400]
check("is_current_stream_capturing" in blk, "capture_node_mamba has CUDA-graph guard (b)")
check("MAMBA_ON_EVICT" in src(r"scheduler\cache.py"), "mamba-on-evict env gate present")

# ---- 4. built-in unit tests ----
for t in ("test_reattach.py", "test_captured_drop.py"):
    p = os.path.join(r"C:\Users\X303\Documents\FreetokenP\tools", t)
    r = subprocess.run([PY, p], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": r"C:\Users\X303\Documents\FreetokenP"})
    check(r.returncode == 0, f"{t} passes", (r.stdout or "")[-200:])

# ---- 5. optional live page accounting (if a server is up) ----
try:
    import json
    import urllib.request
    h = json.load(urllib.request.urlopen("http://127.0.0.1:1919/health", timeout=5))
    check(h.get("status") == "ok", "live /health ok")
    st = json.load(urllib.request.urlopen("http://127.0.0.1:1919/v1/stats", timeout=5))
    print(f"INFO live: kv={st['kv']['used_pages']}/{st['kv']['total_pages']} "
          f"mamba={st['mamba']['used_slots']}/{st['mamba']['total_slots']} "
          f"decode_tps={st['throughput']['decode_tps']}")
except Exception as e:
    print("INFO live checks skipped (no server):", repr(e)[:80])

print("=== RESULT:", "PASS" if ok else "FAIL", "===")
sys.exit(0 if ok else 1)
