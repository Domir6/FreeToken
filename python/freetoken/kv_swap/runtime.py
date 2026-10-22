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


def dump_kv_to_ssd(scheduler) -> dict:
    """Capture resident prefixes into the host tier, then flush it to SSD. Idle-only."""
    cache_manager = getattr(scheduler, "cache_manager", None)
    store = getattr(cache_manager, "host_store", None) if cache_manager is not None else None
    if store is None:
        return {"status": "unsupported", "captured": 0, "persisted": 0}
    captured = 0
    persisted = 0
    budget = os.environ.get("FREETOKEN_KV_DUMP_BYTES", "").strip()
    max_bytes = int(budget) if budget else None
    if hasattr(store, "capture_resident"):
        try:
            captured = store.capture_resident(
                cache_manager.prefix_cache, max_bytes=max_bytes
            )
        except Exception as exc:  # noqa: BLE001 - a failed dump must not block shutdown
            logger.warning(f"host KV capture failed: {exc!r}")
    if hasattr(store, "flush_ssd"):
        try:
            persisted = store.flush_ssd(max_bytes=1 << 62)
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
        logger.info_rank0(store.summary())

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
