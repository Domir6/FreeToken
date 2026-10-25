from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

GIB = 1 << 30

_SESS_LOG = r"C:\AI_Models\logs\kv_boot.log"


def _slog(msg: str) -> None:
    """Session-snapshot trace (capture / restore decision points) -> kv_boot.log.

    Temporary visibility aid: the serve console is a 0-byte transcript, so every silent
    except in this path was invisible. Low-frequency by construction (idle-gated capture,
    cold-start-only restore misses)."""
    try:
        with open(_SESS_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} [sess] {msg}\n")
    except OSError:
        pass

# Staging ring for capture-on-evict (spec v1.1). Hard-coded on purpose: the pinned-memory
# budget is accident-prone (see the 11.6 GB incident), so no env knob is provided.
_MAMBA_STAGING_SLOTS = 4
_MAMBA_DRAIN_CHUNK_BYTES = 16 * 1024 * 1024
# Session snapshots per idle transition (spec ② as revised): the old cap of 1 let the
# largest candidate's dedup-miss starve the smaller ones when their mamba slots aged out
# (capture window lost, next steps degraded to the 8128-token public head).
_SESS_MAX_PER_IDLE = 4


class _StagingTask:
    """One queued mamba snapshot: GPU state already staged into a pinned ring slot."""

    __slots__ = ("key", "prefix", "slot_idx", "patch_target")

    def __init__(self, key: str, prefix: torch.Tensor, slot_idx: int, patch_target=None) -> None:
        self.key = key
        self.prefix = prefix
        self.slot_idx = slot_idx
        self.patch_target = patch_target


class _StagingRing:
    """Fixed-size pinned ring holding in-flight GDN snapshots (one slot ~= 110 MB).

    Slot layout mirrors LinearStatePool buffers minus the slot dimension: conv/recurrent
    plus every dynamic ``slot_states`` tensor. ``alloc`` -> ``copy_in`` happens on the
    scheduler thread inside the eviction hook; the drain step reads the slot and frees it.
    """

    def __init__(self, linear_pool, device: torch.device | None = None,
                 n_slots: int = _MAMBA_STAGING_SLOTS) -> None:
        self.n_slots = n_slots
        self._src_pool = linear_pool
        self._device = device
        self._specs: List[tuple] = []  # (pool_attr, buffer_key, per-snapshot shape, dtype)
        for attr, key in (("conv_states", "conv"), ("recurrent_states", "recurrent")):
            t = getattr(linear_pool, attr, None)
            if isinstance(t, torch.Tensor) and t.dim() >= 2:
                self._specs.append((attr, key, t[:, 0].shape, t.dtype))
        for name, t in (getattr(linear_pool, "slot_states", None) or {}).items():
            if isinstance(t, torch.Tensor) and t.dim() >= 2:
                self._specs.append((name, name, t[:, 0].shape, t.dtype))
        self._bufs: List[Dict[str, torch.Tensor]] = []
        for _ in range(n_slots):
            try:
                self._bufs.append(
                    {
                        key: torch.empty(shape, dtype=dtype, pin_memory=True)
                        for _attr, key, shape, dtype in self._specs
                    }
                )
            except Exception:
                self._bufs = []
                break
        self._busy = [False] * n_slots
        self.slot_bytes = 0
        for buf in self._bufs:
            self.slot_bytes += sum(t.numel() * t.element_size() for t in buf.values())
        # Side copy stream: the D2H runs on the copy engine CONCURRENTLY with whatever the
        # main compute stream is executing. Submitting on the main stream would queue the
        # copy behind every pending prefill/decode kernel and the sync would stall for the
        # whole chunk (the 47-62 ms pre-fix stall was exactly this queueing, not the copy).
        self._side_stream = None
        if self._device is not None and self._device.type == "cuda":
            try:
                self._side_stream = torch.cuda.Stream(self._device)
            except Exception:
                self._side_stream = None

    @property
    def ok(self) -> bool:
        return bool(self._bufs) and all(len(b) == len(self._specs) for b in self._bufs)

    @property
    def busy_slots(self) -> int:
        return sum(self._busy)

    def alloc(self) -> int:
        for i, busy in enumerate(self._busy):
            if not busy:
                self._busy[i] = True
                return i
        return -1

    def copy_in(self, slot_idx: int, mamba_slot: int) -> float:
        """Synchronous D2H of one GDN snapshot into pinned slot ``slot_idx`` (~5 ms).

        The copy runs on a dedicated side stream: DMA proceeds in parallel with the main
        stream's queued kernels, and the CPU-side event sync below waits only for the
        transfer itself. Safe because an evicted slot has no in-flight writer: the main
        stream only ever reads it, and this hook returns before the slot can be handed
        to a new request."""
        t0 = time.monotonic()
        if self._side_stream is not None and self._device.type == "cuda":
            ev_done = torch.cuda.Event()
            with torch.cuda.stream(self._side_stream):
                for attr, key, _shape, _dtype in self._specs:
                    src = getattr(self._src_pool, attr, None)
                    if src is None:
                        src = (getattr(self._src_pool, "slot_states", None) or {})[attr]
                    self._bufs[slot_idx][key].copy_(src[:, mamba_slot])
                ev_done.record(self._side_stream)
            ev_done.synchronize()
        else:
            for attr, key, _shape, _dtype in self._specs:
                src = getattr(self._src_pool, attr, None)
                if src is None:
                    src = (getattr(self._src_pool, "slot_states", None) or {})[attr]
                self._bufs[slot_idx][key].copy_(src[:, mamba_slot])
            if self._device is not None and self._device.type == "cuda":
                torch.cuda.current_stream(self._device).synchronize()
        return (time.monotonic() - t0) * 1000.0

    def take(self, slot_idx: int) -> Dict[str, torch.Tensor]:
        return self._bufs[slot_idx]

    def free(self, slot_idx: int) -> None:
        self._busy[slot_idx] = False


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


@dataclass(eq=False)
class HostEntry:
    tokens: torch.Tensor
    kv_k: Optional[torch.Tensor] = None
    kv_v: Optional[torch.Tensor] = None
    mamba: Optional[dict] = None
    kv_start: int = 0
    cmp_index: Optional[torch.Tensor] = None
    rope_pos: Optional[torch.Tensor] = None
    # Capture kind, mirrored into the SSD meta so the negative cache can be cleared per
    # keyspace: "seg" = radix-node segment (extends a GPU hit), "sess" = whole-session
    # snapshot (cold-start restore, kv_start=0), "mb" = mamba-only entry.
    kind: str = "seg"
    last_used: float = field(default_factory=time.monotonic)

    @property
    def mamba_only(self) -> bool:
        return self.kv_k is None

    @property
    def length(self) -> int:
        return int(self.tokens.numel())

    @property
    def nbytes(self) -> int:
        total = self.tokens.numel() * self.tokens.element_size()
        for t in (self.kv_k, self.kv_v):
            if t is not None:
                total += t.numel() * t.element_size()
        if self.mamba:
            for t in self.mamba.values():
                total += t.numel() * t.element_size()
        for t in (self.cmp_index, self.rope_pos):
            if t is not None:
                total += t.numel() * t.element_size()
        return int(total)


class KVHostStore:
    def __init__(
        self,
        kv_pool,
        linear_pool,
        page_size: int,
        capacity_bytes: int | None = None,
        min_tokens: int | None = None,
        pin_memory: bool | None = None,
    ) -> None:
        self.kv_pool = kv_pool
        self.linear_pool = linear_pool
        self.page_size = int(page_size)
        self.capacity_bytes = (
            _env_int("FREETOKEN_KV_SWAP_HOST_BYTES", 16 * GIB)
            if capacity_bytes is None
            else int(capacity_bytes)
        )
        self.min_tokens = (
            _env_int("FREETOKEN_KV_SWAP_MIN_TOKENS", 512)
            if min_tokens is None
            else int(min_tokens)
        )
        self.pin = (
            _env_bool("FREETOKEN_KV_SWAP_PIN", False) if pin_memory is None else bool(pin_memory)
        )
        self.device = kv_pool.device if kv_pool is not None else torch.device("cuda")
        self._entries: List[HostEntry] = []
        self._by_first: Dict[object, List[HostEntry]] = {}
        self._bytes = 0
        self._lock = threading.RLock()
        self.ssd = None
        self._ssd_flushed: set[str] = set()
        # Keyspaces are segregated by prefix (ruling: never share the raw-sha1 space):
        #   seg:<sha1>  radix-node segment captures      (extends a GPU hit)
        #   mb:<sha1>   mamba-only entries               (two-tier: VRAM -> host)
        #   sess:<sha1> whole-session snapshots          (cold-start restore, kv_start=0)
        self._captured: set[str] = set()
        self._sess_captured: set[str] = set()
        self.persist_dir = os.environ.get("FREETOKEN_KV_PERSIST_DIR") or os.path.join(
            os.getcwd(), "kv_persist"
        )
        if _env_bool("FREETOKEN_KV_PERSIST", True):
            try:
                from freetoken.kv_swap.ssd import SSDStore

                self.ssd = SSDStore(
                    self.persist_dir,
                    _env_int("FREETOKEN_KV_PERSIST_BYTES", 64 * GIB),
                    self.page_size,
                    # archive floor = one page, NOT min_tokens: the seg-extend chain is only
                    # as long as its smallest archived segment -- dropping sub-min_tokens
                    # nodes at spill time broke restore chains mid-way (shallow restores).
                    self.page_size,
                )
                self.ssd._on_drop = self._on_entry_dropped
            except Exception:
                self.ssd = None
        self.signature = self._signature()
        self._mamba_captured: set[str] = set()
        self._mamba_burst_start = 0.0
        self._mamba_burst_count = 0
        self._mamba_burst_ms = 0.0
        self._sess_stats = {"captured": 0, "sess_waiting": 0}
        self._staging_ring = (
            _StagingRing(linear_pool, device=self.device) if linear_pool is not None else None
        )
        self._staging_queue: deque = deque()
        self._staging_stats = {
            "staged": 0,
            "drained": 0,
            "skip_ring_full": 0,
            "dedup_hit": 0,
        }
        self.stats = {
            "saved": 0,
            "saved_bytes": 0,
            "skipped": 0,
            "failed": 0,
            "restored": 0,
            "restored_bytes": 0,
            "host_evicted": 0,
            "ssd_saved": 0,
            "ssd_loaded": 0,
            "mamba_saved": 0,
            "mamba_patched": 0,
            "mamba_over_limit": 0,
        }

    def _on_entry_dropped(self, sha1_hex, kind: str = "seg") -> None:
        """SSD evicted an entry: forget its capture marks so the prefix can be captured again.
        The negative cache is keyspace-segregated (seg:/mb:/sess:)."""
        if not sha1_hex:
            return
        with self._lock:
            if kind == "sess":
                self._sess_captured.discard("sess:" + sha1_hex)
            elif kind == "mb":
                self._mamba_captured.discard("mb:" + sha1_hex)
            else:
                self._captured.discard("seg:" + sha1_hex)

    def _signature(self) -> str:
        """Shape/dtype fingerprint: entries saved by another model or quant are never loaded."""
        parts = [f"page={self.page_size}"]
        buf = getattr(self.kv_pool, "_kv_buffer", None)
        if buf is not None:
            parts.append(f"kv={buf.dtype}|{tuple(buf.shape[1:2])}|{tuple(buf.shape[3:])}")
        for name in ("conv_states", "recurrent_states"):
            t = getattr(self.linear_pool, name, None)
            if isinstance(t, torch.Tensor):
                parts.append(f"{name}={t.dtype}|{tuple(t.shape[1:])}")
        return "|".join(parts)

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._entries)

    def _page_key(self, chunk: torch.Tensor):
        vals = chunk.tolist()
        if self.page_size == 1:
            return vals[0]
        return tuple(vals)

    def _token_prefix(self, node) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        cur = node
        while not cur.is_root():
            parts.append(cur._key.detach().to("cpu", torch.int64))
            cur = cur.parent
        parts.reverse()
        if not parts:
            return torch.empty(0, dtype=torch.int64)
        return torch.cat(parts)

    def _mamba_log(self, n_tokens: int, dt_ms: float, kind: str) -> None:
        now = time.monotonic()
        if now - self._mamba_burst_start > 0.5:
            self._mamba_burst_start = now
            self._mamba_burst_count = 0
            self._mamba_burst_ms = 0.0
        self._mamba_burst_count += 1
        self._mamba_burst_ms += dt_ms
        try:
            with open(
                r"C:\AI_Models\logs\mamba_evict.log", "a", encoding="utf-8"
            ) as _fh:
                _fh.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} {kind} {n_tokens} tok "
                    f"{dt_ms:.1f} ms burst#{self._mamba_burst_count} "
                    f"total {self._mamba_burst_ms:.1f} ms\n"
                )
        except OSError:
            pass
        logger.info(
            f"mamba-on-evict: {kind} {n_tokens} tok in {dt_ms:.1f} ms "
            f"(evict-burst #{self._mamba_burst_count}, total {self._mamba_burst_ms:.1f} ms)"
        )

    def capture_node_mamba(self, node) -> bool:
        """Eviction-path capture (spec v1.1): stage ONLY the GDN snapshot into a pinned
        ring slot (one ~5 ms D2H + sync) and queue it; the idle-gated drain lands it in
        the host store. KV is deliberately NOT copied here -- the idle preload owns that."""
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            with self._lock:
                self.stats["skipped"] += 1
            return False
        if self.linear_pool is None or node.mamba_value is None:
            return False
        ring = self._staging_ring
        if ring is None or not ring.ok:
            return False
        try:
            prefix = self._token_prefix(node)
        except Exception:
            return False
        if prefix.numel() < self.min_tokens:
            return False
        import hashlib

        key = "mb:" + hashlib.sha1(prefix.numpy().tobytes()).hexdigest()
        with self._lock:
            if key in self._mamba_captured:
                self._staging_stats["dedup_hit"] += 1
                return False
            # patch decision is synchronous and cheap: a full-KV entry missing its mamba
            # gets patched by the drain step instead of spawning a duplicate entry
            page_key = self._page_key(prefix[: self.page_size])
            target = None
            for entry in list(self._by_first.get(page_key, ())):
                if entry.kind == "seg" and entry.kv_k is not None and torch.equal(entry.tokens, prefix):
                    if entry.mamba is None:
                        target = entry
                    break
        slot = ring.alloc()
        if slot < 0:
            with self._lock:
                self._staging_stats["skip_ring_full"] += 1
            return False
        try:
            dt_ms = ring.copy_in(slot, node.mamba_value)
        except Exception as exc:  # noqa: BLE001 - capture must never break eviction
            ring.free(slot)
            with self._lock:
                self.stats["failed"] += 1
            logger.warning(f"mamba-on-evict stage failed: {exc!r}")
            return False
        # enqueue-time pre-mark (ruling 6): single scheduler thread, no further recheck
        with self._lock:
            self._mamba_captured.add(key)
            self._staging_queue.append(
                _StagingTask(key=key, prefix=prefix, slot_idx=slot, patch_target=target)
            )
            self._staging_stats["staged"] += 1
        self._mamba_log(int(prefix.numel()), dt_ms, "stage")
        return True

    def capture_evicted_mamba(self, node) -> bool:
        """Hook entry point used by the eviction paths (rate-limited per eviction burst)."""
        limit = _env_int("FREETOKEN_MAMBA_CAPTURE_MAX_PER_EVICT", 8)
        now = time.monotonic()
        if now - self._mamba_burst_start <= 0.5 and self._mamba_burst_count >= limit:
            with self._lock:
                self.stats["mamba_over_limit"] += 1
            return False
        try:
            return self.capture_node_mamba(node)
        except Exception as exc:  # noqa: BLE001 - never break eviction
            with self._lock:
                self.stats["failed"] += 1
            logger.warning(f"mamba-on-evict hook failed: {exc!r}")
            return False

    def _drain_one(self, task: _StagingTask, ring: _StagingRing) -> None:
        t0 = time.monotonic()
        try:
            tensors: Dict[str, torch.Tensor] = {}
            for name, buf in ring.take(task.slot_idx).items():
                out = torch.empty(buf.shape, dtype=buf.dtype)
                flat_out = out.view(-1)
                flat_src = buf.view(-1)
                per = max(1, _MAMBA_DRAIN_CHUNK_BYTES // buf.element_size())
                for s in range(0, flat_out.numel(), per):
                    e = min(s + per, flat_out.numel())
                    flat_out[s:e].copy_(flat_src[s:e])
                tensors[name] = out
            with self._lock:
                if task.patch_target is not None:
                    task.patch_target.mamba = tensors
                    self._ssd_flushed.discard(self._entry_key(task.patch_target))
                    self.stats["mamba_patched"] += 1
                    kind = "patch"
                else:
                    entry = HostEntry(
                        tokens=task.prefix, kv_k=None, kv_v=None, mamba=tensors, kv_start=0
                    )
                    if entry.nbytes <= self.capacity_bytes:
                        # ruling 2: a host-full drain evicts LRU WITHOUT touching the SSD
                        self._make_room_nospill(entry.nbytes)
                        self._insert(entry, count=False)
                        self.stats["mamba_saved"] += 1
                        kind = "store"
                    else:
                        kind = "drop"
            dt_ms = (time.monotonic() - t0) * 1000.0
            if kind == "drop":
                logger.warning(
                    f"mamba staging drop: {task.prefix.numel()} tok exceeds host capacity"
                )
            else:
                self._mamba_log(int(task.prefix.numel()), dt_ms, kind)
        finally:
            ring.free(task.slot_idx)

    def drain_staging(self, max_tasks: int = 4) -> int:
        """Idle-gated drain (spec v1.1 ruling 3): land queued snapshots into the host store
        on the scheduler thread while nobody is waiting; never overlaps decode/prefill."""
        ring = self._staging_ring
        if ring is None or not self._staging_queue:
            return 0
        n = 0
        while n < max_tasks:
            with self._lock:
                if not self._staging_queue:
                    break
                task = self._staging_queue.popleft()
            try:
                self._drain_one(task, ring)
            except Exception as exc:  # noqa: BLE001 - drain must not break the idle gate
                with self._lock:
                    self.stats["failed"] += 1
                logger.warning(f"mamba staging drain failed: {exc!r}")
            finally:
                with self._lock:
                    self._staging_stats["drained"] += 1
            n += 1
        return n

    def drain_all(self) -> int:
        """Ruling 7 (replaces thread join): synchronously empty the queue before any
        SSD flush so a stop-time dump cannot lose staged snapshots."""
        n = 0
        while self._staging_queue:
            batch = self.drain_staging(max_tasks=1 << 30)
            if batch == 0:
                break
            n += batch
        return n

    def _forget_capture_marks(self, entry: HostEntry) -> None:
        """Negative-cache hygiene for HOST-side drops: an entry removed without an SSD
        archive must have its capture mark cleared, or the keyspace dedup blocks re-
        capture forever and the seg-extend chain stays broken (the "missing KV chunk"
        class of restore failure)."""
        key = self._entry_key(entry)
        with self._lock:
            if entry.kind == "sess":
                self._sess_captured.discard(key)
            elif entry.kind == "mb":
                self._mamba_captured.discard(key)
            else:
                self._captured.discard(key)

    def _make_room_nospill(self, nbytes: int, protected: Optional[HostEntry] = None) -> None:
        """Ruling 2: free host space by DROPPING the lowest-ranked entry -- never ``_spill``,
        so the drain path performs zero disk I/O."""
        evicted = {}
        while self._bytes + nbytes > self.capacity_bytes:
            candidates = [e for e in self._entries if e is not protected]
            if not candidates:
                return
            victim = min(candidates, key=self._evict_rank)
            evicted[victim.kind] = evicted.get(victim.kind, 0) + 1
            self._forget_capture_marks(victim)
            self._remove(victim)
            self.stats["host_evicted"] += 1
        if evicted:
            detail = " ".join(f"{k}={v}" for k, v in sorted(evicted.items()))
            _slog(f"evict[drop] {detail} host_bytes={self._bytes}")

    def capture_node(self, node, min_tokens: int | None = None) -> bool:
        floor = self.min_tokens if min_tokens is None else int(min_tokens)
        # a capture-time eviction must not synchronize or allocate device temporaries
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            with self._lock:
                self.stats["skipped"] += 1
            return False
        node_len = int(node.length)
        if node_len < floor:
            with self._lock:
                self.stats["skipped"] += 1
            return False
        if self._bytes >= self.capacity_bytes and self.ssd is None:
            return False
        prefix = self._token_prefix(node)
        if prefix.numel() < floor:
            with self._lock:
                self.stats["skipped"] += 1
            return False
        import hashlib

        key = "seg:" + hashlib.sha1(prefix.numpy().tobytes()).hexdigest()
        with self._lock:
            if key in self._captured:
                return False

        k_cpu = None
        v_cpu = None
        cmp_parts: list[torch.Tensor] = []
        rope_parts: list[torch.Tensor] = []
        has_tiers = hasattr(self.kv_pool, "gather_index_tiers")
        copy_chunk = _env_int("FREETOKEN_KV_SWAP_COPY_CHUNK", 8192)
        try:
            for start in range(0, node_len, copy_chunk):
                chunk = node.value[start : start + copy_chunk]
                k, v = self.kv_pool.gather_kv(chunk)
                if has_tiers:
                    _cmp, _rope = self.kv_pool.gather_index_tiers(chunk)
                    cmp_parts.append(_cmp.to("cpu", copy=True))
                    if _rope is not None:
                        rope_parts.append(_rope.to("cpu", copy=True))
                if k_cpu is None:
                    shape = (k.shape[0], node_len, *k.shape[2:])
                    k_cpu = torch.empty(shape, dtype=k.dtype, pin_memory=self.pin)
                    v_cpu = torch.empty(shape, dtype=v.dtype, pin_memory=self.pin)
                end = start + k.shape[1]
                k_cpu[:, start:end].copy_(k, non_blocking=True)
                v_cpu[:, start:end].copy_(v, non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - a failed capture must not break eviction
            with self._lock:
                self.stats["failed"] += 1
            logger.warning(f"host KV capture failed ({node_len} tokens): {exc!r}")
            return False

        mamba = None
        if node.mamba_value is not None and self.linear_pool is not None:
            mamba = self.linear_pool.snapshot_slot(node.mamba_value)

        entry = HostEntry(
            tokens=prefix,
            kv_k=k_cpu,
            kv_v=v_cpu,
            mamba=mamba,
            kv_start=int(prefix.numel()) - node_len,
            cmp_index=torch.cat(cmp_parts, dim=1) if cmp_parts else None,
            rope_pos=torch.cat(rope_parts, dim=0) if rope_parts else None,
        )
        nbytes = entry.nbytes
        if nbytes > self.capacity_bytes:
            return False
        with self._lock:
            if self.ssd is not None:
                self._make_room(nbytes)
            self._insert(entry)
            self._trim()
            # mark AFTER a successful insert: this line was missing, which made the
            # _captured dedup a no-op and re-captured the same leaves on every idle
            # cycle (the saved-counter churn)
            self._captured.add(key)
        return True

    def _insert(self, entry: HostEntry, count: bool = True) -> None:
        key = self._page_key(entry.tokens[: self.page_size])
        self._entries.append(entry)
        self._by_first.setdefault(key, []).append(entry)
        self._bytes += entry.nbytes
        if count:
            self.stats["saved"] += 1
            self.stats["saved_bytes"] += entry.nbytes

    def _remove(self, entry: HostEntry) -> None:
        try:
            self._entries.remove(entry)
        except ValueError:
            return
        key = self._page_key(entry.tokens[: self.page_size])
        bucket = self._by_first.get(key)
        if bucket is not None:
            try:
                bucket.remove(entry)
            except ValueError:
                pass
            if not bucket:
                del self._by_first[key]
        self._bytes -= entry.nbytes

    def _spill(self, victim: HostEntry) -> None:
        key = self._entry_key(victim)
        if self.ssd is not None:
            try:
                # save() is a cheap stat when the file already exists, and re-writes it when
                # the SSD tier evicted it since the last flush
                if self.ssd.save(victim, self.signature):
                    self.stats["ssd_saved"] += 1
                    self._ssd_flushed.add(key)
            except Exception:
                pass
        self._remove(victim)
        self.stats["host_evicted"] += 1

    def _make_room(self, nbytes: int, protected: Optional[HostEntry] = None) -> None:
        evicted = {}
        while self._bytes + nbytes > self.capacity_bytes:
            candidates = [e for e in self._entries if e is not protected]
            if not candidates:
                return
            victim = min(candidates, key=self._evict_rank)
            evicted[victim.kind] = evicted.get(victim.kind, 0) + 1
            self._spill(victim)
        if evicted:
            detail = " ".join(f"{k}={v}" for k, v in sorted(evicted.items()))
            _slog(f"evict[spill->ssd] {detail} host_bytes={self._bytes}")

    def _trim(self, protected: Optional[HostEntry] = None) -> None:
        self._make_room(0, protected=protected)

    @staticmethod
    def _entry_key(entry: HostEntry) -> str:
        import hashlib

        # kind-prefixed: a seg entry and a sess entry of the same prefix share the sha1 but
        # must occupy distinct _ssd_flushed slots (different payloads)
        return f"{entry.kind}:" + hashlib.sha1(entry.tokens.contiguous().numpy().tobytes()).hexdigest()

    @staticmethod
    def _evict_rank(entry: HostEntry):
        # headroom policy: mamba-only dies first (two-tier by design), then segments, then
        # whole-session snapshots (they are the only cold-start path, die last)
        weight = {"mb": 0, "seg": 1, "sess": 2}.get(entry.kind, 1)
        return (weight, entry.last_used)

    def flush_ssd(self, max_bytes: int | None = None) -> int:
        """Persist resident entries not yet on SSD, up to ``max_bytes`` per call (called from
        the idle loop; a full host tier can be many GiB, so it drains over several cycles)."""
        if self.ssd is None:
            return 0
        budget = (
            _env_int("FREETOKEN_KV_PERSIST_FLUSH_BYTES", 1 * GIB)
            if max_bytes is None
            else int(max_bytes)
        )
        saved = 0
        written = 0
        with self._lock:
            for entry in list(self._entries):
                if written >= budget:
                    break
                # ruling 2: mamba-only entries live two tiers only (VRAM -> host); the SSD
                # tier is reserved for the big full-KV entries
                if entry.mamba_only:
                    continue
                key = self._entry_key(entry)
                if key in self._ssd_flushed:
                    continue
                try:
                    if self.ssd.save(entry, self.signature, write_index=False):
                        self._ssd_flushed.add(key)
                        self.stats["ssd_saved"] += 1
                        saved += 1
                        written += entry.nbytes
                except Exception:
                    pass
            if saved:
                self.ssd.flush_index()
        return saved

    @property
    def dump_flag_path(self) -> str:
        return os.path.join(self.persist_dir, ".dump_request")


    def capture_resident(self, cache, max_bytes: int | None = None) -> int:
        """Capture unlocked radix leaves into the host tier WITHOUT evicting them.

        Used by the shutdown/dump path so the SSD tier learns every prefix still resident in
        GPU KV, even when the pool never fills (98304-token pools rarely do). Repeated calls
        are cheap: captured prefixes are remembered in ``_captured`` and skipped.
        """
        budget = (1 << 62) if max_bytes is None else int(max_bytes)
        leaves = getattr(cache, "_leaves", None)
        if leaves is None:
            return 0
        captured = 0
        used = 0
        seen: set[int] = set()
        for leaf in sorted(leaves(), key=lambda n: -int(n.length)):
            if used >= budget:
                break
            # A node stores only its own KV segment, so walk the whole ancestry (root ->
            # leaf) or a stop-time dump would persist nothing but the last segment.
            chain = []
            node = leaf
            while node is not None and not node.is_root():
                chain.append(node)
                node = getattr(node, "parent", None)
            for item in reversed(chain):
                ident = id(item)
                if ident in seen:
                    continue
                seen.add(ident)
                if item.ref_count != 0 or int(item.length) < self.page_size:
                    continue
                before = self._bytes
                if self.capture_node(item, min_tokens=self.page_size):
                    captured += 1
                    used += max(0, self._bytes - before)
                if used >= budget:
                    break
        return captured

    def capture_sessions(self, cache, max_bytes: int | None = None,
                         exclude_ids: torch.Tensor | None = None) -> int:
        """Whole-session snapshots for cold-start restore (session snapshot spec v1.0).

        Hard cap: _SESS_MAX_PER_IDLE snapshots; the largest uncaptured leaf chains go
        first. ``exclude_ids`` skips the leaf chain that the INCOMING request belongs to
        (pressure-triggered call site): at >80% pool pressure the incoming prefill is
        about to evict the OTHER sessions -- capturing ourselves right before evicting
        them is exactly backwards."""
        if self.linear_pool is None or self._staging_ring is None:
            self._sess_stats["sess_waiting"] = 0
            return 0
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            self._sess_stats["sess_waiting"] = 0
            return 0
        leaves = getattr(cache, "_leaves", None)
        if leaves is None:
            self._sess_stats["sess_waiting"] = 0
            return 0
        budget = (1 << 62) if max_bytes is None else int(max_bytes)
        excl = exclude_ids.detach().to("cpu", torch.int64) if exclude_ids is not None else None
        captured = 0
        waiting = 0
        n_cand = 0
        for leaf in sorted(leaves(), key=lambda n: -int(n.length)):
            if captured >= _SESS_MAX_PER_IDLE:
                waiting += 1
                continue
            if leaf.mamba_value is None or leaf.ref_count != 0:
                continue
            if excl is not None and excl.numel() > 0:
                try:
                    prefix_probe = self._token_prefix(leaf)
                    n = min(int(prefix_probe.numel()), int(excl.numel()))
                    if n > 0 and torch.equal(prefix_probe[:n], excl[:n]):
                        continue  # the incoming request's own chain: it survives this prefill
                except Exception:
                    pass
            n_cand += 1
            if self.capture_session(leaf, budget=budget):
                captured += 1
            else:
                waiting += 1
        self._sess_stats["sess_waiting"] = waiting
        _slog(f"capture scan: candidates={n_cand} captured={captured} waiting={waiting}")
        return captured

    def capture_session(self, leaf, budget: int = 1 << 62) -> bool:
        """Snapshot ONE whole session chain (root->leaf): full-prefix tokens, full KV
        (kv_start=0) and the leaf's GDN snapshot. Before inserting, drop the session's
        seg entries from the host tier (replace, not grow -- ruling 3); their SSD archive,
        if any, stays because it belongs to the rollback layer (ruling 4)."""
        try:
            prefix = self._token_prefix(leaf)
        except Exception as exc:
            _slog(f"capture FAIL token_prefix: {exc!r}")
            return False
        if prefix.numel() < self.min_tokens:
            _slog(f"capture skip: prefix {prefix.numel()} < min_tokens {self.min_tokens}")
            return False
        import hashlib

        key = "sess:" + hashlib.sha1(prefix.numpy().tobytes()).hexdigest()
        with self._lock:
            if key in self._sess_captured:
                _slog(f"capture skip: dedup (len={prefix.numel()}, seen)")
                return False
        entry = self._gather_full_chain(leaf, prefix)
        if entry is None:
            _slog(f"capture skip: gather failed / leaf GDN slot gone (len={prefix.numel()})")
            return False
        if entry.nbytes > min(self.capacity_bytes, budget):
            _slog(f"capture skip: {entry.nbytes} bytes over budget "
                  f"(cap={self.capacity_bytes} budget={budget})")
            return False
        with self._lock:
            if key in self._sess_captured:
                return False
            self._drop_session_segs(prefix)
            if self.ssd is not None:
                self._make_room(entry.nbytes)
            self._insert(entry)
            self._trim()
            self._sess_captured.add(key)
            self._sess_stats["captured"] += 1
        _slog(f"captured sess len={prefix.numel()} bytes={entry.nbytes} "
              f"host_bytes={self._bytes}")
        return True

    def _gather_full_chain(self, leaf, prefix: torch.Tensor) -> Optional[HostEntry]:
        """Gather the KV of the whole root->leaf chain into one contiguous CPU buffer.
        Returns None when the tree shifts mid-walk or the leaf has no live GDN slot
        (a snapshot without the linear state could not be restored into a valid session)."""
        chain = []
        node = leaf
        while node is not None and not node.is_root():
            chain.append(node)
            node = getattr(node, "parent", None)
        if not chain:
            return None
        chain.reverse()
        total = sum(int(n.length) for n in chain)
        if total != int(prefix.numel()):
            return None
        has_tiers = hasattr(self.kv_pool, "gather_index_tiers")
        copy_chunk = _env_int("FREETOKEN_KV_SWAP_COPY_CHUNK", 8192)
        k_cpu = v_cpu = None
        cmp_parts: List[torch.Tensor] = []
        rope_parts: List[torch.Tensor] = []
        try:
            offset = 0
            for node in chain:
                node_len = int(node.length)
                for start in range(0, node_len, copy_chunk):
                    chunk = node.value[start : start + copy_chunk]
                    k, v = self.kv_pool.gather_kv(chunk)
                    if has_tiers:
                        _cmp, _rope = self.kv_pool.gather_index_tiers(chunk)
                        cmp_parts.append(_cmp.to("cpu", copy=True))
                        if _rope is not None:
                            rope_parts.append(_rope.to("cpu", copy=True))
                    if k_cpu is None:
                        shape = (k.shape[0], total, *k.shape[2:])
                        k_cpu = torch.empty(shape, dtype=k.dtype, pin_memory=self.pin)
                        v_cpu = torch.empty(shape, dtype=v.dtype, pin_memory=self.pin)
                    end = offset + k.shape[1]
                    k_cpu[:, offset:end].copy_(k, non_blocking=True)
                    v_cpu[:, offset:end].copy_(v, non_blocking=True)
                    offset = end
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - snapshot must never break the idle gate
            with self._lock:
                self.stats["failed"] += 1
            logger.warning(f"session chain gather failed ({total} tokens): {exc!r}")
            return None
        mamba = None
        if self.linear_pool is not None and leaf.mamba_value is not None:
            try:
                mamba = self.linear_pool.snapshot_slot(leaf.mamba_value)
            except Exception:
                mamba = None
        if mamba is None:
            # Tombstoned leaf (GDN slot reclaimed but KV still in the tree): the chain's
            # linear state exists as an mb entry (mamba-on-evict archived it at reclaim
            # time). Without this, eviction marks the whole session un-snapshotable and
            # every revisit degrades to the shared-head restore (8128).
            try:
                mb = self.lookup(prefix, min_len=0, require_kv=False)
                if mb is not None and mb.mamba is not None and mb.length == prefix.numel():
                    mamba = mb.mamba
            except Exception:
                mamba = None
        if mamba is None:
            return None
        return HostEntry(
            tokens=prefix,
            kv_k=k_cpu,
            kv_v=v_cpu,
            mamba=mamba,
            kv_start=0,
            kind="sess",
            cmp_index=torch.cat(cmp_parts, dim=1) if cmp_parts else None,
            rope_pos=torch.cat(rope_parts, dim=0) if rope_parts else None,
        )

    def _drop_session_segs(self, prefix: torch.Tensor) -> None:
        """RETIRED per user ruling (2026-10-25): snapshots are an immutable version
        history -- a superseded snapshot is roll-back material ("revert to before the
        compaction"), so nothing is dropped at capture time. Retention belongs to the
        host LRU (24G, kind-ranked mb->seg->sess) with SSD as the archive tier; SSD
        eviction clears the negative cache (re-capturable). Kept as a no-op for the
        call site's shape."""
        return

    def lookup_session(self, input_ids) -> Optional[HostEntry]:
        """Cold-start path (cached_len=0): exact whole-session entry, host tier first,
        then the SSD archive. Hit is refreshed (LRU) but NOT removed -- repeat visits
        keep the entry hot until it is evicted or consumed by take()."""
        if input_ids.numel() == 0:
            return None
        ids = input_ids.detach().to("cpu", torch.int64)
        key = self._page_key(ids[: self.page_size])
        with self._lock:
            bucket = list(self._by_first.get(key, ()))
        best = None
        prefix_hit = False
        for entry in bucket:
            if entry.kind != "sess" or entry.kv_start != 0:
                continue
            L = int(entry.tokens.numel())
            if L == int(ids.numel()) and torch.equal(ids, entry.tokens):
                best = entry
                break
            if L > int(ids.numel()) and torch.equal(entry.tokens[: ids.numel()], ids):
                # User "revert-then-resend" flow: the request is a PREFIX of a stored
                # snapshot. Serve a truncated view (zero-copy KV slice) instead of
                # skipping -- the head's KV and GDN state are bit-identical (GDN is a
                # deterministic recurrence over the prefix).
                _n = int(ids.numel())
                best = HostEntry(
                    tokens=ids.clone(),
                    kv_k=entry.kv_k[:, :_n] if entry.kv_k is not None else None,
                    kv_v=entry.kv_v[:, :_n] if entry.kv_v is not None else None,
                    mamba=entry.mamba,
                    kv_start=0,
                    kind="sess",
                    cmp_index=(entry.cmp_index[:, : _n // int(getattr(self.kv_pool, "index_ratio", 1))]
                               if entry.cmp_index is not None else None),
                    rope_pos=(entry.rope_pos[:_n] if entry.rope_pos is not None else None),
                )
                prefix_hit = True
                break
        if best is not None:
            best.last_used = time.monotonic()
            if prefix_hit:
                _slog(f"restore PREFIX hit: request {_n} of snapshot {L} (truncated view)")
            else:
                _slog(f"restore HOST hit: len={best.length} of {ids.numel()} asked")
            return best
        # Miss diagnostics: locate the first divergence against the longest sess candidate,
        # so prefix-stability bugs identify themselves (the equal() above failed for a
        # reason -- find WHERE).
        longest = None
        for entry in bucket:
            if entry.kind != "sess" or entry.kv_start != 0 or entry.length > ids.numel():
                continue
            if longest is None or entry.length > longest.length:
                longest = entry
        if longest is not None:
            ne = entry_tokens = longest.tokens
            ni = ids[: ne.numel()]
            neq = ne != ni
            idx = int(torch.nonzero(neq)[0]) if bool(neq.any()) else -1
            _slog(f"restore DIVERGE: sess len={ne.numel()} diverges at token {idx} "
                  f"(asked {ids.numel()}); around divergence ids={ni[max(0, idx - 4): idx + 4].tolist()} "
                  f"entry={ne[max(0, idx - 4): idx + 4].tolist() if idx >= 0 else 'exact-prefix-equal'}")
        if self.ssd is None:
            _slog(f"restore MISS: no sess entry for {ids.numel()} tok (ssd off)")
            return None
        try:
            entry = self.ssd.lookup_session(ids, self.signature)
        except Exception as exc:
            _slog(f"restore ssd lookup raised: {exc!r}")
            entry = None
        if entry is None or entry.nbytes > self.capacity_bytes:
            _slog(f"restore MISS: no sess entry for {ids.numel()} tok (host+ssd)")
            return None
        _slog(f"restore SSD hit: len={entry.length} of {ids.numel()} asked")
        with self._lock:
            self._insert(entry, count=False)
            self._ssd_flushed.add(self._entry_key(entry))
            self._trim(protected=entry)
        return entry

    def lookup(
        self, input_ids, min_len: int = 0, require_kv: bool = True
    ) -> Optional[HostEntry]:
        if input_ids.numel() == 0:
            return None
        ids = input_ids.detach().to("cpu", torch.int64)
        key = self._page_key(ids[: self.page_size])
        with self._lock:
            bucket = list(self._by_first.get(key, ()))
        best = None
        for entry in bucket:
            if require_kv and entry.kv_k is None:
                continue
            length = entry.length
            if length <= min_len or length > ids.numel():
                continue
            # entries may begin before the caller's hit; the overlapping KV is skipped on copy
            if entry.kv_start > min_len:
                continue
            if torch.equal(ids[:length], entry.tokens):
                if best is None or length > best.length:
                    best = entry
        if best is not None:
            best.last_used = time.monotonic()
            return best
        if self.ssd is None:
            return None
        try:
            entry = self.ssd.lookup(ids, min_len, self.signature, require_kv=require_kv)
        except Exception as exc:
            entry = None
        if entry is None or entry.nbytes > self.capacity_bytes:
            return None
        with self._lock:
            self._insert(entry, count=False)
            self._ssd_flushed.add(self._entry_key(entry))
            self.stats["ssd_loaded"] += 1
            self._trim(protected=entry)
        return entry

    def take(self, entry: HostEntry) -> None:
        """Record a restore. The entry STAYS in the host tier: sessions re-visit, and the
        host LRU (24G budget, kind-ranked eviction) owns retention -- dropping the copy
        here forced every revisit down the SSD slow path (torch.load of ~2 GB), which is
        what "restore reads from SSD every time" traced back to."""
        with self._lock:
            self.stats["restored"] += 1
            self.stats["restored_bytes"] += entry.nbytes

    def summary(self) -> str:
        text = (
            f"host_kv entries={len(self._entries)} bytes={self._bytes / GIB:.2f}GiB "
            f"saved={self.stats['saved']} restored={self.stats['restored']} "
            f"skipped={self.stats['skipped']} failed={self.stats['failed']} "
            f"host_evicted={self.stats['host_evicted']}"
        )
        if self.ssd is not None:
            text += (
                f" | {self.ssd.summary()} | ssd_promoted={self.stats['ssd_saved']}"
                f" ssd_loaded={self.stats['ssd_loaded']}"
            )
        st = self._staging_stats
        ring = self._staging_ring
        inflight = ring.busy_slots if ring is not None else 0
        ring_free = (ring.n_slots - inflight) if ring is not None else 0
        text += (
            f" | mamba_staging staged={st['staged']} drained={st['drained']} "
            f"skip_ring_full={st['skip_ring_full']} dedup_hit={st['dedup_hit']} "
            f"ring_free={ring_free} inflight={inflight}"
        )
        text += f" sess_waiting={self._sess_stats['sess_waiting']}"
        return text


def create_host_store(kv_pool, linear_pool, page_size: int) -> Optional[KVHostStore]:
    if not _env_bool("FREETOKEN_KV_SWAP", True):
        return None
    if kv_pool is None or not hasattr(kv_pool, "gather_kv"):
        return None
    return KVHostStore(kv_pool, linear_pool, page_size)
