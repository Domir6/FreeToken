"""Unit test for CacheManager._try_reattach_mamba (the pure-mamba re-attach path).

Constructs the three elements the production path needs, then calls the method directly:
  1. KV present in the radix tree   (node with live value/page indices)
  2. mamba tombstoned               (node.mamba_value is None, as evict_mamba leaves it)
  3. host holds a mamba-only entry  (kv_k is None, mamba is not None, same tokens)

Asserts the node gets a fresh slot re-attached, the restored snapshot equals the stored one
(max|d| == 0), and the host entry is consumed -- with zero KV copy.
"""
import os
import sys

import torch

sys.path.insert(0, r"C:\Users\X303\Documents\FreetokenP")

from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache, HybridCacheHandle  # noqa: E402
from freetoken.scheduler.cache import CacheManager  # noqa: E402
from freetoken.kv_swap.manager import HostEntry  # noqa: E402

PAGE = 64
N = 256  # 4 pages


class FakePool:
    def __init__(self):
        self._free = [7, 8, 9]
        self.restored = {}

    @property
    def num_free_slots(self):
        return len(self._free)

    def alloc(self, n=1):
        out = [self._free.pop() for _ in range(n)]
        return out

    def free(self, slots):
        self._free.extend(slots)

    def restore_slot(self, slot, data):
        self.restored[slot] = {k: v.clone() for k, v in data.items()}

    def snapshot_slot(self, slot):
        return self.restored[slot]


class FakeStore:
    def __init__(self, entry):
        self._entry = entry
        self.taken = 0

    def lookup(self, ids, min_len=0, require_kv=True):
        e = self._entry
        if require_kv and e.kv_k is None:
            return None
        if e.length > ids.numel():
            return None
        if not torch.equal(ids[: e.length], e.tokens):
            return None
        return e

    def take(self, entry):
        self.taken += 1


def main():
    torch.manual_seed(0)
    pc = HybridRadixCache(torch.device("cpu"), PAGE)
    ids = torch.arange(1000, 1000 + N, dtype=torch.int64)
    kv = torch.arange(N, dtype=torch.int32)

    # 1. KV in the tree (fresh node carries a snapshot)
    ins_len, exist = pc.insert(ids, kv, 3)
    assert ins_len == 0 and exist is False, (ins_len, exist)  # insert() returns pre-existing len
    m = pc.match_prefix(ids)
    node = m.node
    assert node.mamba_value == 3

    # 2. tombstone it exactly like evict_mamba does (KV + node survive)
    pc._free_node_mamba(node, [])
    assert node.mamba_value is None
    assert pc.match_prefix(ids).cached_len == 0, "tombstoned node must stop resuming"

    # 3. host has a mamba-only entry for the same tokens
    snap = {
        "conv": torch.randn(2, 5),
        "recurrent": torch.randn(3, 4, 6),
    }
    entry = HostEntry(tokens=ids.clone(), kv_k=None, kv_v=None, mamba=snap, kv_start=0)
    assert entry.mamba_only

    # build a minimal CacheManager around the real radix cache
    cm = CacheManager.__new__(CacheManager)
    cm.prefix_cache = pc
    cm.linear_state_pool = FakePool()
    cm.host_store = FakeStore(entry)
    cm.device = torch.device("cpu")
    cm.page_size = PAGE
    cm.is_hybrid = True
    cm.is_swa = False
    cm.swa_pool = None
    cm.page_table = torch.zeros(1, N, dtype=torch.int32)

    handle_before = HybridCacheHandle(0, pc.root, pc.empty)
    new_handle = CacheManager._try_reattach_mamba(cm, ids, handle_before)

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok = ok and cond

    check(new_handle is not None, "reattach returned a handle (not None)")
    if new_handle is None:
        print("=== RESULT: FAIL ===")
        sys.exit(1)
    check(node.mamba_value is not None, f"node re-attached to slot {node.mamba_value}")
    check(new_handle.cached_len == N, f"handle resumes at {new_handle.cached_len} == {N}")

    slot = node.mamba_value
    got = cm.linear_state_pool.restored.get(slot)
    check(got is not None, "pool.restore_slot was called for the new slot")
    if got is not None:
        dmax = 0.0
        for k in snap:
            dmax = max(dmax, (got[k].float() - snap[k].float()).abs().max().item())
        check(dmax == 0.0, f"max|d| = {dmax} (expected 0) -- snapshot identical")
    check(cm.host_store.taken == 1, "host mamba-only entry was consumed (take)")

    # and the node is resumable again
    m2 = pc.match_prefix(ids)
    check(m2.cached_len == N, f"match_prefix resumes again at {m2.cached_len}")

    print("=== RESULT:", "PASS" if ok else "FAIL", "===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
