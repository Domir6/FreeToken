"""Spec v1.1 unit tests for the mamba staging ring (CPU + mocks, no server needed).

Covers section 4.1:
  ring-full skip without blocking / HostEntry field parity / ledger identity
  staged == drained + skip_ring_full + inflight / drain_all empties everything
  patch_existing path / flush_ssd skips mamba-only / drain LRU never touches the SSD.
Run:  python tools/test_staging.py
"""
import hashlib
import os
import shutil
import sys
import tempfile
import time

import torch

sys.path.insert(0, r"C:\Users\X303\Documents\FreetokenP")
os.environ.setdefault("FREETOKEN_MAMBA_CAPTURE_MAX_PER_EVICT", "100")

from freetoken.kv_swap.manager import (  # noqa: E402
    KVHostStore,
    _MAMBA_STAGING_SLOTS,
    _StagingRing,
)

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
        out = {
            "conv": self.conv_states[:, slot].detach().to("cpu", copy=True),
            "recurrent": self.recurrent_states[:, slot].detach().to("cpu", copy=True),
        }
        for name, t in self.slot_states.items():
            out[name] = t[:, slot].detach().to("cpu", copy=True)
        return out


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

    @property
    def length(self):
        return int(self._key.numel())

    def is_root(self):
        return self.parent is None


def make_node(start, mamba_value=None, seg=100):
    root = FakeNode(torch.empty(0, dtype=torch.int64))
    n1 = FakeNode(torch.arange(start, start + seg, dtype=torch.int64), root)
    n2 = FakeNode(torch.arange(start + seg, start + 2 * seg, dtype=torch.int64), n1, mamba_value)
    return n2


def make_store(capacity_bytes=1 << 30, min_tokens=64, with_ssd=None):
    store = KVHostStore(
        FakeKVPool(),
        FakeLinearPool(),
        64,
        capacity_bytes=capacity_bytes,
        min_tokens=min_tokens,
    )
    store._mamba_log = lambda *a, **k: None  # keep production logs clean
    if with_ssd is not None:
        store.ssd = with_ssd
        store.ssd._on_drop = store._on_entry_dropped
    return store


def main():
    check(_MAMBA_STAGING_SLOTS == 4, "ring size is the hard-coded constant 4")

    # T1: ring full -> 5th capture skipped, no blocking, counters correct
    store = make_store()
    ring = store._staging_ring
    check(ring.ok and ring.n_slots == 4, "ring built with 4 pinned slots")
    t0 = time.monotonic()
    results = [store.capture_node_mamba(make_node(1000 * i, mamba_value=i)) for i in range(5)]
    dt = time.monotonic() - t0
    check(results == [True, True, True, True, False], "4 stages succeed, the 5th is skipped")
    check(store._staging_stats["staged"] == 4, "staged counter == 4")
    check(store._staging_stats["skip_ring_full"] == 1, "skip_ring_full counter == 1")
    check(dt < 5.0, f"no blocking on ring-full (total {dt * 1000:.1f} ms)")

    # T2: slot ledger identity staged == drained + inflight (skip never occupies a slot)
    check(
        store._staging_stats["staged"]
        == store._staging_stats["drained"] + ring.busy_slots,
        "slot ledger identity holds mid-flight (staged = drained + inflight)",
    )
    check(store._staging_stats["skip_ring_full"] == 1, "skip counter is independent")

    # T3: drain produces a HostEntry with field parity vs the mamba-only path
    n_drained = store.drain_staging(max_tasks=10)
    check(n_drained == 4, "drain lands all 4 queued tasks")
    entries = store._entries
    check(len(entries) == 4, "4 mamba-only entries in the host store")
    snap_keys = set(FakeLinearPool().snapshot_slot(0).keys())
    ok_fields = True
    for e in entries:
        ok_fields &= (
            e.kv_k is None
            and e.kv_v is None
            and e.kv_start == 0
            and set(e.mamba.keys()) == snap_keys
            and e.nbytes > 0
        )
    check(ok_fields, "HostEntry fields identical to mamba-only path (kv_k=None, 4-key mamba)")
    e0 = entries[0]
    expect_key = "seg:" + hashlib.sha1(e0.tokens.numpy().tobytes()).hexdigest()
    check(store._entry_key(e0) == expect_key, "entry sha1 key matches its token prefix (kind-prefixed)")
    check(
        store._staging_stats["staged"]
        == store._staging_stats["drained"] + ring.busy_slots,
        "slot ledger identity holds after drain",
    )
    check(ring.busy_slots == 0, "all ring slots returned after drain")

    # T4: drain_all empties everything (ruling 7, replaces thread join)
    for i in range(3):
        store.capture_node_mamba(make_node(20000 + 1000 * i, mamba_value=i))
    check(ring.busy_slots == 3, "3 slots busy before drain_all")
    store.drain_all()
    check(len(store._staging_queue) == 0 and ring.busy_slots == 0, "drain_all leaves no backlog")

    # T5: patch_existing path -- an existing full-KV entry missing mamba gets patched
    store2 = make_store()
    node_kv = make_node(5000)  # mamba_value=None -> capture_node stores KV without mamba
    check(store2.capture_node(node_kv), "capture_node stores a full-KV entry")
    kv_entry = store2._entries[-1]
    check(kv_entry.kv_k is not None and kv_entry.mamba is None, "KV entry initially mamba-less")
    same_prefix_node = make_node(5000, mamba_value=3)
    check(store2.capture_node_mamba(same_prefix_node), "stage against existing prefix succeeds")
    before = len(store2._entries)
    store2.drain_all()
    check(len(store2._entries) == before, "patch does not spawn a duplicate entry")
    check(kv_entry.mamba is not None, "drain patched mamba onto the existing entry")
    check(store2.stats["mamba_patched"] == 1, "mamba_patched counter == 1")

    # T6a: flush_ssd skips mamba-only entries, persists full-KV ones
    tmp = tempfile.mkdtemp(prefix="kv_staging_test_")
    try:
        from freetoken.kv_swap.ssd import SSDStore

        store3 = make_store()
        store3.ssd = SSDStore(tmp, 1 << 30, 64, 64)
        store3.ssd._on_drop = store3._on_entry_dropped
        check(store3.capture_node_mamba(make_node(7000, mamba_value=1)), "stage mamba-only")
        check(store3.capture_node(make_node(8000)), "capture full-KV")
        store3.drain_all()
        saved_before = store3.ssd.stats["saved"]
        store3.flush_ssd()
        flushed_kv = any(e.kv_k is not None for e in store3._entries)
        check(
            store3.ssd.stats["saved"] - saved_before == (1 if flushed_kv else 0),
            "flush_ssd persists only the full-KV entry (skips mamba-only)",
        )

        # T6b: drain LRU eviction performs zero SSD writes (ruling 2)
        # 53 KB: the ~52 KB full-KV entry fits, a second entry does not -> drain must evict
        store4 = make_store(capacity_bytes=53_000)
        store4.ssd = SSDStore(tmp, 1 << 30, 64, 64)
        store4.ssd._on_drop = store4._on_entry_dropped
        check(store4.capture_node(make_node(6000)), "seed a full-KV entry")
        store4.flush_ssd()  # give the seed entry an SSD copy, like the idle loop would
        ssd_saved_before = store4.ssd.stats["saved"]
        evicted_before = store4.stats["host_evicted"]
        check(store4.capture_node_mamba(make_node(6500, mamba_value=2)), "stage one more")
        store4.drain_all()
        check(
            store4.stats["host_evicted"] > evicted_before,
            "drain evicted the LRU entry to make room",
        )
        check(
            store4.ssd.stats["saved"] == ssd_saved_before,
            "drain eviction performed no SSD writes",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("=== RESULT:", "PASS" if not FAILURES else "FAIL", "===")
    sys.exit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
