"""Runtime hooks for the compiled Scheduler.

From FreeToken 0.2.0 on, ``freetoken/scheduler/scheduler.py`` ships as a Cython
extension (``scheduler.cp312-win_amd64.pyd``), so the KV host/SSD tier cannot be wired
into the source any more. This module patches the exported ``Scheduler`` class after it
is imported: the tier is created right after ``__init__``, the idle dump/preload hooks
run after ``run_when_idle``, and ``shutdown`` captures resident prefixes before the
engine tears the pools down.

``freetoken/scheduler/__init__.py`` calls :func:`install` once the class is importable.
"""

from __future__ import annotations

import os
import time

from freetoken.utils import init_logger

logger = init_logger(__name__)

_installed = False
_idle_count = 0
_BOOT_LOG = os.environ.get("FREETOKEN_KV_BOOT_LOG", r"C:\AI_Models\logs\kv_boot.log")


def _boot(msg: str) -> None:
    """Low-frequency wiring trace (startup / first idle gate); the console is not captured
    by Start-Transcript, so keep a small file to debug the tier on compiled schedulers."""
    try:
        with open(_BOOT_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except OSError:
        pass


def _mem_probe() -> str:
    """Free RAM (Win32 GlobalMemoryStatusEx) and free VRAM (torch), both pure reads.

    Attached to the existing 30 s statistics line only — no extra sampling point, no
    thread, no env. 17-t/s case proved a single axis (RAM) points at the wrong culprit,
    so both numbers must appear together."""
    free_ram = 0.0
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            free_ram = st.ullAvailPhys / (1024 ** 3)
    except Exception:
        pass
    free_vram = 0.0
    try:
        # nvidia-smi's "memory.free" is the physical-free axis the 17 t/s case demanded
        # (foreign graphical processes show up here); cudaMemGetInfo reports how much THIS
        # process may still cudaMalloc, which is ~0 whenever the engine fills its quota
        # and would read 0.00 forever. Single-GPU deployment: first line is the card.
        import subprocess

        out = subprocess.run(
            [r"C:\Windows\System32\nvidia-smi.exe",
             "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        if out and out[0]:
            free_vram = float(out[0]) / 1024.0
    except Exception:
        try:
            import torch

            free_vram = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        except Exception:
            pass
    return f"mem freeRAM={free_ram:.1f}GiB freeVRAM={free_vram:.2f}GiB"


def dump_kv_to_ssd(scheduler) -> dict:
    """Capture resident prefixes into the host tier, then flush it to SSD. Idle-only."""
    cache_manager = getattr(scheduler, "cache_manager", None)
    store = getattr(cache_manager, "host_store", None) if cache_manager is not None else None
    if store is None:
        return {"status": "unsupported", "captured": 0, "persisted": 0}
    # ruling 7 (replaces the thread join): land every staged snapshot BEFORE the SSD flush
    # so a stop-time dump cannot lose ring-resident captures
    if hasattr(store, "drain_all"):
        try:
            store.drain_all()
        except Exception as exc:  # noqa: BLE001 - never block the dump on staging
            logger.warning(f"mamba staging drain_all failed: {exc!r}")
    captured = 0
    persisted = 0
    # Default budget 1 GiB: a long-lived server fills the host tier (16 GiB), and an
    # unbounded dump would re-capture every leaf and re-write the whole tier to SSD,
    # stalling shutdown for minutes (CTRL_C / stop script then miss their graceful
    # window and fall back to hard-kill). The idle loop already flushes 1 GiB per
    # 30 s cycle, so the un-flushed residue at stop time is small.
    budget = os.environ.get("FREETOKEN_KV_SWAP_DUMP_BYTES", "").strip() or str(1 << 30)
    max_bytes = int(budget)
    if hasattr(store, "capture_resident"):
        try:
            captured = store.capture_resident(
                cache_manager.prefix_cache, max_bytes=max_bytes
            )
        except Exception as exc:  # noqa: BLE001 - a failed dump must not block shutdown
            logger.warning(f"host KV capture failed: {exc!r}")
    if hasattr(store, "flush_ssd"):
        try:
            persisted = store.flush_ssd(max_bytes=max_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"host KV SSD flush failed: {exc!r}")
    return {
        "status": "ok",
        "captured": captured,
        "persisted": persisted,
        "summary": store.summary() if hasattr(store, "summary") else "",
    }


def _idle_hooks(scheduler) -> None:
    cache_manager = getattr(scheduler, "cache_manager", None)
    if cache_manager is None:
        return
    try:
        cache_manager._kv_diag("idle")
    except Exception:  # noqa: BLE001 - diagnostics only
        pass
    try:
        cache_manager.check_integrity()
    except Exception as exc:  # noqa: BLE001 - page accounting gap is logged, not fatal
        logger.warning(f"cache integrity check: {exc!r}")
        _boot(f"integrity FAILED: {exc!r}")
    # Tree-side page accounting vs the pool's used_pages (/v1/stats): a growing gap
    # between the two is a dirty-eviction (VRAM residue) signature.
    try:
        pc = getattr(cache_manager, "prefix_cache", None)
        if pc is not None:
            tree_pages = (pc.full_evictable + pc.full_protected) // max(
                getattr(cache_manager, "page_size", 1), 1)
            mamba_tree = getattr(pc, "mamba_evictable_size", 0) + getattr(pc, "mamba_protected", 0)
            _boot(f"integrity: tree_pages={tree_pages} mamba_tree_slots={mamba_tree}")
    except Exception as _e:
        _boot(f"integrity probe failed: {_e!r}")

    store = getattr(cache_manager, "host_store", None)
    global _idle_count
    _idle_count += 1
    if _idle_count <= 3:
        _boot(f"idle#{_idle_count}: host_store={'present' if store is not None else 'None'}")
    if store is None:
        return
    now = time.monotonic()
    if now - getattr(scheduler, "_host_store_log_ts", 0.0) > 30.0:
        scheduler._host_store_log_ts = now
        _summary = store.summary() + " | " + _mem_probe()
        logger.info_rank0(_summary)
        # Mirror the host-tier counters to a file: they are the only readable signal for
        # "who is doing the work" (idle-preload coverage vs eviction), since the scheduler
        # subprocess stdout is not captured by Start-Transcript.
        try:
            with open(
                r"C:\AI_Models\logs\kv_preload.log", "a", encoding="utf-8"
            ) as _fh:
                _fh.write(
                    time.strftime("%Y-%m-%d %H:%M:%S") + " " + _summary + "\n"
                )
        except OSError:
            pass

    # SSD persistence is explicit: the stop script writes <persist_dir>/.dump_request and
    # sends one more token; the idle gate below turns that flag into a dump.
    flag = getattr(store, "dump_flag_path", None)
    if flag and os.path.exists(flag):
        try:
            result = dump_kv_to_ssd(scheduler)
            logger.info_rank0(
                f"host KV dump: status={result['status']} captured={result['captured']} "
                f"persisted={result['persisted']} | {result.get('summary', '')}"
            )
        finally:
            try:
                os.remove(flag)
            except OSError:
                pass

    # Idle-gated staging drain (spec v1.1 ruling 3): land queued mamba snapshots into the
    # host store while nobody is waiting; bounded per idle cycle so it never competes for
    # bandwidth with the preload below. Runs BEFORE the preload: the ring is small and the
    # queue must not back up into skip_ring_full.
    drain_max = os.environ.get("FREETOKEN_MAMBA_DRAIN_MAX_PER_IDLE", "4").strip()
    if hasattr(store, "drain_staging"):
        try:
            store.drain_staging(max_tasks=int(drain_max) if drain_max else 4)
        except Exception as exc:  # noqa: BLE001 - background optimization
            logger.warning(f"mamba staging drain failed: {exc!r}")

    # Whole-session snapshots (session snapshot spec v1.0): cold-start restore for
    # entirely evicted sessions. One session per idle transition, hard cap.
    sess_on = os.environ.get("FREETOKEN_KV_SESS_SNAPSHOT", "1").strip().lower()
    if sess_on not in ("", "0", "false", "no", "off") and not getattr(
        scheduler, "_sess_capture_disabled", False
    ):
        if now - getattr(scheduler, "_sess_capture_ts", 0.0) > 1.0:
            scheduler._sess_capture_ts = now
            try:
                n_sess = store.capture_sessions(cache_manager.prefix_cache)
                if n_sess:
                    logger.info_rank0(
                        f"session snapshot: captured={n_sess} "
                        f"waiting={store._sess_stats['sess_waiting']}"
                    )
            except Exception as exc:  # noqa: BLE001 - background optimization
                if "out of memory" in str(exc).lower():
                    scheduler._sess_capture_disabled = True
                    _BOOT_LOG_MSG = f"session snapshot disabled after CUDA OOM: {exc!r}"
                    logger.warning(_BOOT_LOG_MSG)
                    try:
                        with open(_BOOT_LOG, "a", encoding="utf-8") as fh:
                            fh.write(f"{time.strftime('%H:%M:%S')} [sess] {_BOOT_LOG_MSG}\n")
                    except OSError:
                        pass
                else:
                    _BOOT_LOG_MSG = f"session snapshot failed: {exc!r}"
                    logger.warning(_BOOT_LOG_MSG)
                    try:
                        with open(_BOOT_LOG, "a", encoding="utf-8") as fh:
                            fh.write(f"{time.strftime('%H:%M:%S')} [sess] {_BOOT_LOG_MSG}\n")
                    except OSError:
                        pass

    # Pre-stage evictable prefixes into the host tier while nobody is waiting: a later
    # request then evicts them for free instead of paying a synchronous D2H on decode.
    preload = os.environ.get("FREETOKEN_KV_SWAP_PRELOAD_ON_IDLE", "1").strip().lower()
    if preload not in ("", "0", "false", "no", "off") and not getattr(
        scheduler, "_idle_capture_disabled", False
    ):
        if now - getattr(scheduler, "_idle_capture_ts", 0.0) > 1.0:
            scheduler._idle_capture_ts = now
            try:
                moved = store.capture_resident(
                    cache_manager.prefix_cache,
                    max_bytes=int(
                        os.environ.get("FREETOKEN_KV_SWAP_IDLE_BYTES", str(1 << 30))
                    ),
                )
                if moved:
                    logger.info_rank0(
                        f"host KV idle preload: captured={moved} | {store.summary()}"
                    )
            except Exception as exc:  # noqa: BLE001 - background optimization
                if "out of memory" in str(exc).lower():
                    scheduler._idle_capture_disabled = True
                    logger.warning(
                        f"host KV idle preload disabled after CUDA OOM: {exc!r}"
                    )
                else:
                    logger.warning(f"host KV idle preload failed: {exc!r}")


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True

    from freetoken.scheduler.scheduler import Scheduler

    orig_init = Scheduler.__init__
    orig_idle = Scheduler.run_when_idle
    orig_shutdown = Scheduler.shutdown

    def _init(self, config):
        orig_init(self, config)
        cache_manager = getattr(self, "cache_manager", None)
        if cache_manager is None:
            attrs = [a for a in dir(self) if "cache" in a.lower()]
            _boot(f"init: no cache_manager; cache-like attrs={attrs}")
            logger.warning("KV swap: Scheduler exposes no cache_manager; tier not installed")
            return
        if getattr(cache_manager, "host_store", None) is not None:
            _boot("init: host_store already present")
            return
        engine = getattr(self, "engine", None)
        kv_pool = getattr(engine, "kv_cache", None)
        linear_pool = getattr(engine, "linear_state_pool", None)
        _boot(
            f"init: cache_manager={type(cache_manager).__name__} engine={type(engine).__name__} "
            f"kv_pool={type(kv_pool).__name__ if kv_pool is not None else None} "
            f"linear_pool={type(linear_pool).__name__ if linear_pool is not None else None} "
            f"page_size={getattr(cache_manager, 'page_size', None)}"
        )
        if kv_pool is None:
            _boot("init: engine has no kv_cache; tier not installed")
            logger.warning("KV swap: engine exposes no kv_cache; tier not installed")
            return
        from freetoken.kv_swap.manager import KVHostStore

        _boot(
            "init: env KV_SWAP=" + repr(os.environ.get("FREETOKEN_KV_SWAP"))
            + " KV_PERSIST=" + repr(os.environ.get("FREETOKEN_KV_PERSIST"))
            + " PERSIST_DIR=" + repr(os.environ.get("FREETOKEN_KV_PERSIST_DIR"))
            + " cwd=" + os.getcwd()
        )
        disabled = os.environ.get("FREETOKEN_KV_SWAP", "1").strip().lower()
        if disabled in ("0", "false", "off", "no"):
            _boot("init: FREETOKEN_KV_SWAP=0; tier not installed")
            return
        cache_manager.kv_pool = kv_pool
        cache_manager.host_store = KVHostStore(kv_pool, linear_pool, cache_manager.page_size)
        cache_manager._attach_evict_hook()
        store = cache_manager.host_store
        _boot(
            f"init: tier enabled persist_dir={getattr(store, 'persist_dir', None)} "
            f"min_tokens={getattr(store, 'min_tokens', None)}"
        )
        logger.info(f"KV swap tier enabled: {store.summary()}")

    def _idle(self):
        orig_idle(self)
        try:
            _idle_hooks(self)
        except Exception as exc:  # noqa: BLE001 - never break the idle gate
            logger.warning(f"host KV idle hooks failed: {exc!r}")

    def _shutdown(self):
        try:
            result = dump_kv_to_ssd(self)
            logger.info_rank0(
                f"host KV shutdown dump: captured={result['captured']} "
                f"persisted={result['persisted']}"
            )
        except Exception as exc:  # noqa: BLE001 - never block shutdown on the SSD tier
            logger.warning(f"host KV shutdown dump failed: {exc!r}")
        orig_shutdown(self)

    Scheduler.__init__ = _init
    Scheduler.run_when_idle = _idle
    Scheduler.shutdown = _shutdown
    Scheduler._dump_kv_to_ssd = lambda self: dump_kv_to_ssd(self)
    from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache

    _orig_free_node_mamba = HybridRadixCache._free_node_mamba

    if not getattr(_orig_free_node_mamba, "_kv_mamba_wrapped", False):

        def _free_node_mamba(self, node, out):

            cap = getattr(self, "_kv_mamba_capture", None)

            if cap is not None and node.mamba_value is not None:

                try:

                    cap(node)

                except Exception:  # noqa: BLE001 - capture must never break eviction

                    pass

            return _orig_free_node_mamba(self, node, out)

        _free_node_mamba._kv_mamba_wrapped = True

        HybridRadixCache._free_node_mamba = _free_node_mamba

    logger.debug("KV swap mamba-on-evict hook installed on HybridRadixCache")


    logger.debug("KV swap runtime hooks installed on Scheduler")
