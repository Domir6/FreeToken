from .config import SchedulerConfig
from .scheduler import Scheduler

# KV host/SSD tier: 0.2.0 ships scheduler.py as a Cython extension, so the hooks live in
# kv_swap/runtime.py and are installed here once the class is importable.
from freetoken.kv_swap.runtime import install as _install_kv_swap

_install_kv_swap()

__all__ = ["Scheduler", "SchedulerConfig"]
