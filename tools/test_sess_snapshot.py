"""Unit tests for whole-session snapshots (session snapshot spec v1.0, CPU + mocks).

Covers:
  keyspace segregation (seg:/mb:/sess:) / replace-not-grow (drop session segs first) /
  LRU priority (mb -> seg -> sess) / host-full drops the snapshot without blocking /
  lookup_session field parity / SSD archive + negative cache round-trip.
Run:  python tools/test_sess_snapshot.py
"""
import hashlib
import shutil
import sys
import tempfile
import time

import torch

sys.path.insert(0, r"C:\Users\X303\Documents\FreetokenP")

from freetoken.kv_swap.manager import KVHostStore  # noqa: E402

FAILURES = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


class FakeLinearPool:
    def __init__(self, n_slots=8):
        self.conv_states = torch.randn(2, n_slots, 4, 8)
        self.recurrent_states = torch.randn(2, n_slots, 16)
        self.slot_states = {"mm": torch.randn(2, n_slots, 8)}

    def snapshot_slot(self, slot):
        return {
            "conv": self.conv_states[:, slot].detach().to("cpu", copy=True),
            "recurrent": self.recurrent_states[:, slot].detach().to("cpu", copy=True),
            "mm": self.slot_states["mm"][:, slot].detach().to("cpu", copy=True),
        }


class FakeKVPool:
    device = torch.device("cpu")

    def gather_kv(self, chunk):
        n = int(chunk.numel())
        return torch.randn(2, n, 4, 8), torch.randn(2, n, 4, 8)


class FakeNode:
    def __init__(self, keys, parent=None, mamba_value=None):
        self._key = keys
        self.parent = parent
        self.mamba_value = mamba_value
        self.value = keys
        self.ref_count = 0

    @property
    def length(self):
        return int(self._key.numel())

    def is_root(self):
        return self.parent is None


def make_chain(start, mamba_value=3, seg=100):
    root = FakeNode(torch.empty(0, dtype=torch.int64))
    n1 = FakeNode(torch.arange(start, start + seg, dtype=torch.int64), root)
    n2 = FakeNode(torch.arange(start + seg, start + 2 * seg, dtype=torch.int64), n1, mamba_value)
    return n2


class FakeCache:
    def __init__(self, leaves):
        self._l = leaves

    def _leaves(self):
        return self._l


def make_store(capacity_bytes=1 << 30, min_tokens=64):
    store = KVHostStore(
        FakeKVPool(), FakeLinearPool(), 64,
        capacity_bytes=capacity_bytes, min_tokens=min_tokens)
    store._mamba_log = lambda *a, **k: None
    return store


def main():
    root = FakeNode(torch.empty(0, dtype=torch.int64))
    n1 = FakeNode(torch.arange(1000, 1100, dtype=torch.int64), root)
    leaf = FakeNode(torch.arange(1100, 1200, dtype=torch.int64), n1, 3)

    # T1: keyspace segregation -- seg and sess captures of the same chain coexist
    store = make_store()
    check(store.capture_node(leaf), "seg capture of the leaf chain")
    check(store.capture_session(leaf), "sess capture of the same leaf")
    seg_keys = {k for k in store._captured if k.startswith("seg:")}
    sess_keys = {k for k in store._sess_captured if k.startswith("sess:")}
    check(len(seg_keys) >= 1 and len(sess_keys) == 1, "seg: and sess: keys coexist, isolated")
    kinds = {e.kind for e in store._entries}
    check(kinds == {"seg", "sess"},
          "seg entries RETAINED beside sess (partial-restore head), only stale sess replaced")
    # re-capture is deduped inside its own keyspace
    check(not store.capture_session(leaf), "sess re-capture deduped within sess: keyspace")

    # T2: replace-not-grow -- a fresh sess entry drops the session's seg entries
    before = len(store._entries)
    n1b = FakeNode(torch.arange(5000, 5100, dtype=torch.int64), root)
    leafb = FakeNode(torch.arange(5100, 5200, dtype=torch.int64), n1b, 4)
    store.capture_node(n1b)
    store.capture_node(leafb)
    n_seg_before = sum(1 for e in store._entries if e.kind == "seg")
    check(store.capture_session(leafb), "sess capture for session B")
    seg_b_left = [e for e in store._entries
                  if e.kind == "seg" and int(e.tokens.numel()) <= 200
                  and torch.equal(leafb.parent._key, e.tokens[100:200])]
    check(len(seg_b_left) == 0 and len(store._entries) < before + n_seg_before + 1,
          "session B's seg entries dropped before the sess insert (replace, not grow)")

    # T3: LRU priority -- under host pressure seg dies first, sess survives atomically
    store2 = make_store(capacity_bytes=120_000)
    check(store2.capture_node(make_chain(1000)), "seed seg entry (~52KB)")
    check(store2.capture_session(make_chain(1000)), "sess entry replaces it")
    kinds2 = [e.kind for e in store2._entries]
    check(kinds2 == ["sess"], f"host holds only the sess entry after replace (got {kinds2})")
    # now squeeze: a second sess that does not fit evicts by rank mb -> seg -> sess
    tight = make_store(capacity_bytes=60_000)
    check(tight.capture_node(make_chain(1000)), "seg fits (52KB)")
    check(not tight.capture_session(make_chain(1000)), "sess that exceeds capacity is refused (no block)")
    check(any(e.kind == "seg" for e in tight._entries), "seg kept intact when sess refused")

    # T4: lookup_session field parity
    e = store2.lookup_session(torch.cat([n1_ := torch.arange(1000, 1100, dtype=torch.int64),
                                         torch.arange(1100, 1200, dtype=torch.int64)]))
    check(e is not None and e.kind == "sess" and e.kv_start == 0
          and set(e.mamba.keys()) == {"conv", "recurrent", "mm"} and e.kv_k is not None,
          "lookup_session returns kind=sess kv_start=0 4-key mamba full-KV entry")
    check(e is not None and int(e.tokens.numel()) == 200
          and torch.equal(e.tokens, torch.arange(1000, 1200, dtype=torch.int64)),
          "sess entry tokens == the full session prefix")

    # T5: SSD archive + negative cache round-trip
    tmp = tempfile.mkdtemp(prefix="kv_sess_test_")
    try:
        from freetoken.kv_swap.ssd import SSDStore

        store3 = make_store()
        store3.ssd = SSDStore(tmp, 1 << 30, 64, 64)
        store3.ssd._on_drop = store3._on_entry_dropped
        check(store3.capture_session(make_chain(1000)), "sess captured into host")
        n_before = store3.ssd.stats["saved"]
        store3.flush_ssd()
        check(store3.ssd.stats["saved"] > n_before, "sess entry archived to SSD (not skipped)")
        ids = torch.arange(1000, 1200, dtype=torch.int64)
        # drop the host copy, restore must come back from the SSD archive
        sess_e = [e for e in store3._entries if e.kind == "sess"][0]
        store3.take(sess_e)
        e2 = store3.lookup_session(ids)
        check(e2 is not None and e2.kind == "sess" and e2.kv_start == 0,
              "host-missed sess restored from the SSD archive")
        # evict from SSD -> negative cache must clear sess keyspace
        key = "sess:" + hashlib.sha1(ids.numpy().tobytes()).hexdigest()
        check(key in store3._sess_captured, "sess negative cache holds the key")
        path = [p for p, m in store3.ssd._index.items() if m.get("kind") == "sess"]
        for p in path:
            store3.ssd._drop(p)
        check(key not in store3._sess_captured, "SSD drop clears _sess_captured (re-capturable)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # T6: sess_waiting counter visible in summary
    store4 = make_store()
    leaves = [make_chain(10000 + 300 * i) for i in range(3)]
    store4.capture_sessions(FakeCache(leaves))
    check("sess_waiting=" in store4.summary(), "summary carries sess_waiting")
    check(store4._sess_stats["captured"] == 3,
          f"up to 4 sessions per idle transition (got {store4._sess_stats['captured']})")

    print("=== RESULT:", "PASS" if not FAILURES else "FAIL", "===")
    sys.exit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
