from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

GIB = 1 << 30


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
        self._captured: set[str] = set()
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
                    self.min_tokens,
                )
                self.ssd._on_drop = self._on_entry_dropped
            except Exception:
                self.ssd = None
        self.signature = self._signature()
        self._mamba_captured: set[str] = set()
        self._mamba_burst_start = 0.0
        self._mamba_burst_count = 0
        self._mamba_burst_ms = 0.0
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

    def _on_entry_dropped(self, sha1_hex) -> None:
        """SSD evicted an entry: forget its capture marks so the prefix can be captured again."""
        if not sha1_hex:
            return
        with self._lock:
            self._captured.discard(sha1_hex)
            self._mamba_captured.discard(sha1_hex)

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
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            with self._lock:
                self.stats["skipped"] += 1
            return False
        """Eviction-path capture: persist ONLY the GDN snapshot (one D2H), never KV.

        The node's KV needs no copy here -- evict_mamba tombstones the node (KV stays in the
        tree) and evict_full leaves KV to the idle preload. Uses the synchronous
        snapshot_slot() only; never gather_kv, so it avoids the D2H+sync stall that made
        capture-on-evict unusable."""
        if self.linear_pool is None or node.mamba_value is None:
            return False
        try:
            prefix = self._token_prefix(node)
        except Exception:
            return False
        if prefix.numel() < self.min_tokens:
            return False
        import hashlib

        key = hashlib.sha1(prefix.numpy().tobytes()).hexdigest()
        with self._lock:
            if key in self._mamba_captured:
                return False
        page_key = self._page_key(prefix[: self.page_size])
        with self._lock:
            bucket = list(self._by_first.get(page_key, ()))
        target = None
        for entry in bucket:
            if entry.kv_k is not None and torch.equal(entry.tokens, prefix):
                if entry.mamba is None:
                    target = entry
                break
        t0 = time.monotonic()
        try:
            snapshot = self.linear_pool.snapshot_slot(node.mamba_value)
        except Exception as exc:  # noqa: BLE001 - capture must never break eviction
            with self._lock:
                self.stats["failed"] += 1
            logger.warning(f"mamba-on-evict capture failed: {exc!r}")
            return False
        dt_ms = (time.monotonic() - t0) * 1000.0
        if target is not None:
            with self._lock:
                target.mamba = snapshot
                self._mamba_captured.add(key)
                self._ssd_flushed.discard(self._entry_key(target))
                self.stats["mamba_patched"] += 1
            self._mamba_log(int(prefix.numel()), dt_ms, "patch")
            return True
        entry = HostEntry(
            tokens=prefix, kv_k=None, kv_v=None, mamba=snapshot, kv_start=0
        )
        if entry.nbytes > self.capacity_bytes:
            return False
        with self._lock:
            if key in self._mamba_captured:
                return False
            self._mamba_captured.add(key)
            if self.ssd is not None:
                self._make_room(entry.nbytes)
            self._insert(entry, count=False)
            self.stats["mamba_saved"] += 1
            self._trim()
        if self.ssd is not None:
            with self._lock:
                self._ssd_flushed.discard(self._entry_key(entry))
        self._mamba_log(int(prefix.numel()), dt_ms, "store")
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

        key = hashlib.sha1(prefix.numpy().tobytes()).hexdigest()
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
        while self._bytes + nbytes > self.capacity_bytes:
            candidates = [e for e in self._entries if e is not protected]
            if not candidates:
                return
            self._spill(min(candidates, key=lambda e: e.last_used))

    def _trim(self, protected: Optional[HostEntry] = None) -> None:
        self._make_room(0, protected=protected)

    @staticmethod
    def _entry_key(entry: HostEntry) -> str:
        import hashlib

        return hashlib.sha1(entry.tokens.contiguous().numpy().tobytes()).hexdigest()

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
        with self._lock:
            self._remove(entry)
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
        return text


def create_host_store(kv_pool, linear_pool, page_size: int) -> Optional[KVHostStore]:
    if not _env_bool("FREETOKEN_KV_SWAP", True):
        return None
    if kv_pool is None or not hasattr(kv_pool, "gather_kv"):
        return None
    return KVHostStore(kv_pool, linear_pool, page_size)
