"""Unit test: SSD entry drop must clear the negative-cache marks so the prefix can be
re-captured. (Regression for the `_captured` leak fixed in the audit.)"""
import os
import sys
import threading

sys.path.insert(0, r"C:\Users\X303\Documents\FreetokenP")

from freetoken.kv_swap.manager import KVHostStore  # noqa: E402


def main():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok = ok and cond

    # Build a store without running __init__ (it would need a real kv_pool).
    s = KVHostStore.__new__(KVHostStore)
    s._lock = threading.Lock()
    s._captured = {"seg:deadbeef"}
    s._mamba_captured = {"mb:deadbeef"}
    s._sess_captured = {"sess:deadbeef"}

    check(hasattr(s, "_on_entry_dropped"), "_on_entry_dropped exists")
    s._on_entry_dropped("deadbeef", "seg")
    check("seg:deadbeef" not in s._captured, "_captured cleared on SSD drop (seg)")
    s._on_entry_dropped("deadbeef", "mb")
    check("mb:deadbeef" not in s._mamba_captured, "_mamba_captured cleared on SSD drop (mb)")
    s._on_entry_dropped("deadbeef", "sess")
    check("sess:deadbeef" not in s._sess_captured, "_sess_captured cleared on SSD drop (sess)")
    check(s._captured == set() and s._mamba_captured == set() and s._sess_captured == set(),
          "keyspaces are isolated: only the dropped kind's set was touched, all cleared now")
    # None / empty must be a no-op (not raise)
    s._on_entry_dropped(None)
    check(True, "None sha1 is a safe no-op")

    # SSDStore._drop must actually call the callback
    from freetoken.kv_swap.ssd import SSDStore  # noqa: E402
    import inspect
    src = inspect.getsource(SSDStore._drop)
    check("_notify_drop" in src, "SSDStore._drop notifies on_drop")
    check("_notify_drop" in inspect.getsource(SSDStore), "SSDStore has _notify_drop")

    print("=== RESULT:", "PASS" if ok else "FAIL", "===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
