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

# ---- 3b. mamba staging (spec v1.1) static presence ----
check("_MAMBA_STAGING_SLOTS = 4" in mgr, "staging ring size is the hard-coded constant 4")
check("FREETOKEN_KV_SWAP_MAMBA_ON_EVICT" in src(r"scheduler\cache.py"), "staging switch env present")
rt = src(r"kv_swap\runtime.py")
dump_body = rt.split("def dump_kv_to_ssd(")[1].split("def ")[0]
check(
    dump_body.find("drain_all") < dump_body.find("flush_ssd"),
    "drain_all runs before flush_ssd in the dump path (ruling 7)",
)
check("FREETOKEN_MAMBA_DRAIN_MAX_PER_IDLE" in rt, "idle-gated drain wired into the idle hook")
check("mamba_only" in mgr and "if entry.mamba_only:\n                    continue" in mgr,
      "flush_ssd skips mamba-only entries (two-tier semantics)")

# ---- 3c. session snapshot (spec v1.0) static presence ----
check("def capture_sessions(" in mgr, "manager.capture_sessions present")
check("def lookup_session(" in mgr, "manager.lookup_session present")
check("def _drop_session_segs(" in mgr, "manager._drop_session_segs present (replace-not-grow)")
check('"sess:" + hashlib' in mgr or '"sess:"' in mgr, "sess keyspace prefix present")
check("_sess_captured" in mgr, "sess negative cache (keyspace isolation)")
check("lookup_session(input_ids) if cached_len == 0" in src(r"scheduler\cache.py"),
      "cache.attempt_host_restore has the cold-start sess branch")
check("FREETOKEN_KV_SESS_SNAPSHOT" in rt, "session snapshot switch wired in runtime")
check("sess_waiting" in mgr, "sess_waiting counter in summary")

# ---- 4. built-in unit tests ----
for t in ("test_reattach.py", "test_captured_drop.py", "test_staging.py", "test_sess_snapshot.py"):
    p = os.path.join(r"C:\Users\X303\Documents\FreetokenP\tools", t)
    r = subprocess.run([PY, p], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": r"C:\Users\X303\Documents\FreetokenP"})
    check(r.returncode == 0, f"{t} passes", (r.stdout or "")[-200:])

# ---- 4b. staging dynamic invariants (spec v1.1: ledger / rebuild / exit, CPU mocks) ----
try:
    import torch  # noqa: F401

    sys.path.insert(0, r"C:\Users\X303\Documents\FreetokenP")
    os.environ.setdefault("FREETOKEN_MAMBA_CAPTURE_MAX_PER_EVICT", "100")
    from freetoken.kv_swap.manager import KVHostStore  # noqa: E402

    class _LP:
        conv_states = torch.randn(2, 8, 4, 8)
        recurrent_states = torch.randn(2, 8, 16)
        slot_states = {"mm": torch.randn(2, 8, 8)}

    class _KP:
        device = torch.device("cpu")

    class _Nd:
        def __init__(self, keys, parent, mv):
            self._key = keys
            self.parent = parent
            self.mamba_value = mv
            self.value = keys

        @property
        def length(self):
            return int(self._key.numel())

        def is_root(self):
            return self.parent is None

    root = _Nd(torch.empty(0, dtype=torch.int64), None, None)
    n1 = _Nd(torch.arange(1000, 1100, dtype=torch.int64), root, 0)
    n2 = _Nd(torch.arange(1100, 1200, dtype=torch.int64), n1, 1)
    store = KVHostStore(_KP(), _LP(), 64, capacity_bytes=1 << 30, min_tokens=64)
    store._mamba_log = lambda *a, **k: None
    ring = store._staging_ring
    check(store.capture_node_mamba(n1), "staging: stage #1 accepted")
    check(store.capture_node_mamba(n2), "staging: stage #2 accepted")
    check(
        store._staging_stats["staged"] == store._staging_stats["drained"] + ring.busy_slots,
        "staging ledger identity holds (staged = drained + inflight)",
    )
    # rebuild simulation: pools are replaced wholesale, the ring holds CPU copies and the
    # ledger must stay flat, then drain to zero without leaks (spec v1.1 ruling, item 3)
    store.kv_pool = _KP()
    store.linear_pool = _LP()
    drained = store.drain_all()
    check(drained == 2, f"rebuild-invariant: drain_all lands both tasks (got {drained})")
    check(
        store._staging_stats["staged"] == store._staging_stats["drained"] + ring.busy_slots
        and ring.busy_slots == 0
        and len(store._staging_queue) == 0,
        "rebuild-invariant: ledger flat, no slot leak, no queue hang after drain",
    )
    check(len(store._entries) == 2 and all(e.mamba_only for e in store._entries),
          "drain produced exactly the two mamba-only entries")
except Exception as exc:  # noqa: BLE001
    check(False, "staging dynamic invariants raise-free", repr(exc)[:160])

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
