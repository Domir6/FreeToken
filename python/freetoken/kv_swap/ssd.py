"""SSD tier for :class:`~freetoken.kv_swap.manager.KVHostStore`.

One file per captured prefix under ``<base>/<signature>/entries/`` plus an ``index.json``
listing their length, first-page bucket and content hash. A prefix that no longer fits in the
pinned host tier is written here instead of being dropped, so it survives a restart; ``lookup``
falls back to the index when the host tier misses.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Dict, List, Optional

import torch

GIB = 1 << 30


def _sha1(tokens: torch.Tensor) -> str:
    return hashlib.sha1(tokens.contiguous().numpy().tobytes()).hexdigest()


class SSDStore:
    def __init__(
        self,
        base_dir: str,
        capacity_bytes: int,
        page_size: int,
        min_tokens: int = 0,
    ) -> None:
        self.base_dir = os.path.abspath(base_dir)
        self.capacity_bytes = int(capacity_bytes)
        self.page_size = int(page_size)
        self.min_tokens = int(min_tokens)
        self._dir: Optional[str] = None
        self._index: Dict[str, dict] = {}
        self._by_first: Dict[object, List[str]] = {}
        self._bytes = 0
        self._lock = threading.RLock()
        self.stats = {"saved": 0, "loaded": 0, "dropped": 0, "evicted": 0, "failed": 0}

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def _page_key(self, tokens: torch.Tensor):
        vals = tokens[: self.page_size].tolist()
        return vals[0] if self.page_size == 1 else tuple(vals)

    def _signature_dir(self, signature: str) -> str:
        return os.path.join(self.base_dir, hashlib.sha1(signature.encode()).hexdigest()[:16])

    def _ensure_dir(self, signature: str) -> None:
        if self._dir is not None:
            return
        root = self._signature_dir(signature)
        os.makedirs(os.path.join(root, "entries"), exist_ok=True)
        self._dir = root
        self._load_index()

    def _index_path(self) -> str:
        return os.path.join(self._dir, "index.json")

    def _load_index(self) -> None:
        path = self._index_path()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return
        entries = raw.get("entries") or {}
        for name, meta in entries.items():
            full = os.path.join(self._dir, "entries", name)
            if not os.path.exists(full):
                continue
            self._index[full] = meta
            self._by_first.setdefault(tuple(meta["first"]), []).append(full)
            self._bytes += int(meta["bytes"])

    def _write_index(self) -> None:
        if self._dir is None:
            return
        entries = {}
        for full, meta in self._index.items():
            entries[os.path.basename(full)] = meta
        tmp = self._index_path() + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"entries": entries}, fh)
            os.replace(tmp, self._index_path())
        except OSError:
            self.stats["failed"] += 1

    def save(self, entry, signature: str, write_index: bool = True) -> bool:
        if entry.length < max(self.min_tokens, 1):
            return False
        with self._lock:
            self._ensure_dir(signature)
            digest = _sha1(entry.tokens)
            name = f"{digest[:20]}_{int(entry.kv_start)}_{entry.length}.pt"
            path = os.path.join(self._dir, "entries", name)
            if path in self._index and os.path.exists(path):
                # Same token path -> same file name. A mamba-only capture must never clobber a
                # full KV entry, but a full KV entry DOES upgrade a mamba-only one.
                old = self._index[path]
                if not (old.get("mamba_only") and entry.kv_k is not None):
                    old["atime"] = time.time()
                    self.stats["saved"] += 1
                    if write_index:
                        self._write_index()
                    return True
            payload = {
                "tokens": entry.tokens,
                "kv_k": entry.kv_k,
                "kv_v": entry.kv_v,
                "mamba": entry.mamba,
                "kv_start": int(entry.kv_start),
                "cmp_index": entry.cmp_index,
                "rope_pos": entry.rope_pos,
            }
            tmp = path + ".tmp"
            try:
                torch.save(payload, tmp)
                os.replace(tmp, path)
                size = os.path.getsize(path)
            except OSError:
                self.stats["failed"] += 1
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                return False
            first = list(entry.tokens[: self.page_size].tolist())
            meta = {
                "first": first,
                "length": entry.length,
                "kv_start": int(entry.kv_start),
                "sha1": digest,
                "bytes": int(size),
                "atime": time.time(),
                "mamba_only": entry.kv_k is None,
            }
            if path not in self._index:
                self._bytes += int(size)
                self._by_first.setdefault(tuple(first), []).append(path)
            else:
                old = self._index[path]
                self._bytes += int(size) - int(old["bytes"])
            self._index[path] = meta
            self.stats["saved"] += 1
            self._evict()
            if write_index:
                self._write_index()
            return True

    def flush_index(self) -> None:
        with self._lock:
            self._write_index()

    def _evict(self) -> None:
        while self._bytes > self.capacity_bytes and self._index:
            path = min(self._index, key=lambda p: self._index[p]["atime"])
            self._drop(path)
            self.stats["evicted"] += 1

    def _notify_drop(self, meta: dict) -> None:
        cb = getattr(self, "_on_drop", None)
        if cb is not None:
            try:
                cb(meta.get("sha1"))
            except Exception:
                pass

    def _drop(self, path: str) -> None:
        meta = self._index.pop(path, None)
        if meta is None:
            return
        self._notify_drop(meta)
        self._bytes -= int(meta["bytes"])
        bucket = self._by_first.get(tuple(meta["first"]))
        if bucket is not None:
            try:
                bucket.remove(path)
            except ValueError:
                pass
            if not bucket:
                del self._by_first[tuple(meta["first"])]
        try:
            os.remove(path)
        except OSError:
            pass

    def lookup(
        self, ids: torch.Tensor, min_len: int, signature: str, require_kv: bool = True
    ):
        from freetoken.kv_swap.manager import HostEntry

        with self._lock:
            self._ensure_dir(signature)
            key = tuple(ids[: self.page_size].tolist())
            bucket = list(self._by_first.get(key, ()))
        best_meta = None
        best_path = None
        for path in bucket:
            meta = self._index.get(path)
            if meta is None:
                continue
            if require_kv and meta.get("mamba_only"):
                continue
            length = int(meta["length"])
            if length <= min_len or length > ids.numel():
                continue
            # entries whose KV begins at or before the caller's hit may still extend it; the
            # overlap is tolerated because the copy path skips it (an entry captured deeper
            # down the tree has kv_start > min_len and still cannot stand in for a shallow hit)
            if int(meta.get("kv_start", min_len)) > min_len:
                continue
            if best_meta is None or length > int(best_meta["length"]):
                if _sha1(ids[:length]) != meta["sha1"]:
                    continue
                best_meta, best_path = meta, path
        if best_meta is None:
            return None
        try:
            data = torch.load(best_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, ValueError):
            self.stats["failed"] += 1
            with self._lock:
                self._drop(best_path)
            return None
        kv_k = data.get("kv_k")
        if require_kv and kv_k is None:
            self.stats["dropped"] += 1
            return None
        entry = HostEntry(
            tokens=data["tokens"].to(torch.int64).contiguous(),
            kv_k=kv_k,
            kv_v=data.get("kv_v"),
            mamba=data.get("mamba"),
            kv_start=int(data["kv_start"]),
            cmp_index=data.get("cmp_index"),
            rope_pos=data.get("rope_pos"),
        )
        if entry.length != int(best_meta["length"]) or not torch.equal(
            ids[: entry.length], entry.tokens
        ):
            self.stats["dropped"] += 1
            return None
        with self._lock:
            best_meta["atime"] = time.time()
            self.stats["loaded"] += 1
            self._write_index()
        return entry

    def summary(self) -> str:
        return (
            f"ssd_kv entries={len(self._index)} bytes={self._bytes / GIB:.2f}GiB "
            f"saved={self.stats['saved']} loaded={self.stats['loaded']} "
            f"evicted={self.stats['evicted']} dropped={self.stats['dropped']} "
            f"failed={self.stats['failed']}"
        )
