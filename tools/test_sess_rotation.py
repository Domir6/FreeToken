"""Session-snapshot rotation acceptance (spec v1.0): two ~45k sessions, two laps.

87040-token pool cannot hold both (2x45k = 90k) -> by lap 2 each session returns with
cached_len=0 and must come back via the WHOLE-SESSION restore (<=3s), not a ~25s full
prefill. With FREETOKEN_KV_RESTORE_VERIFY=1 the restore_verify.log must show
skip=0 (whole-chain, not a segment extend) and max|d|=0 on every tier.

Run while the server is up:  python tools/test_sess_rotation.py
"""
import io
import json
import re
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:1919"
VERIFY_LOG = r"C:\AI_Models\logs\restore_verify.log"
N_SESSIONS = 2
TARGET_TOKENS = 45_000
RESTORE_BUDGET_S = 3.0
FAILURES = []


def check(cond, msg, extra=""):
    print(("PASS " if cond else "FAIL ") + msg + (("  <- " + extra) if extra and not cond else ""))
    if not cond:
        FAILURES.append(msg)


def chat(text, timeout=900):
    body = json.dumps(
        {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": 1}
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    t0 = time.monotonic()
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    return resp, time.monotonic() - t0


def make_text(n_words, salt):
    return " ".join("%s%05x" % (salt, (i * 2654435761) & 0xFFFFF) for i in range(n_words))


def tail(path):
    try:
        return io.open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return []


def main():
    tag = "ss" + sys.argv[1] if len(sys.argv) > 1 else "ssA"
    probe = {"model": "x", "max_tokens": 1,
             "messages": [{"role": "user", "content": make_text(1000, tag + "0")}]}
    body = json.dumps(probe).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    resp = json.load(urllib.request.urlopen(req, timeout=300))
    got = resp["usage"]["prompt_tokens"]
    n_words = int(1000 * TARGET_TOKENS / got)
    print(f"calibrated: {got} tok @ 1000 words -> {n_words} words per session")

    sessions = [make_text(n_words, tag + str(i)) for i in range(N_SESSIONS)]
    v_before = len(tail(VERIFY_LOG))

    print(f"lap 1 (cold builds + host fills):")
    lap1 = []
    for i, text in enumerate(sessions):
        _, dur = chat(text)
        lap1.append(dur)
        print(f"  s{i + 1}: {dur:.1f}s")
    print("idle window for the snapshot captures ...")
    time.sleep(15)

    print(f"lap 2 (both sessions return with cached_len=0 -> whole-session restore):")
    lap2 = []
    for i, text in enumerate(sessions):
        _, dur = chat(text)
        lap2.append(dur)
        print(f"  s{i + 1}: {dur:.1f}s")

    worst = max(lap2)
    check(worst <= RESTORE_BUDGET_S,
          f"[SS-1] lap-2 whole-session restore <= {RESTORE_BUDGET_S}s (worst {worst:.2f}s; "
          f"lap-1 cold was {max(lap1):.1f}s)")

    new_v = tail(VERIFY_LOG)[v_before:]
    kv_lines = [l for l in new_v if l.startswith("KV restore verify")]
    mb_lines = [l for l in new_v if l.startswith("mamba restore verify")]
    whole = [l for l in kv_lines if re.search(r"skip=0 add=\d+ ", l)]
    bad = [l for l in kv_lines if "max|dK|=0" not in l or "max|dV|=0" not in l]
    bad += [l for l in mb_lines if "max|d|=0" not in l]
    if kv_lines or mb_lines:
        check(not bad, f"[SS-2] {len(kv_lines)} KV + {len(mb_lines)} mamba verify lines all max|d|=0",
              "; ".join(bad[:2]))
        check(len(whole) > 0, f"[SS-3] whole-chain signature present (skip=0): {len(whole)} of {len(kv_lines)}")
        for l in (kv_lines[-1:], mb_lines[-1:]):
            for x in l:
                print("  LOG: " + x)
    else:
        print("INFO [SS-2/3] no verify lines (server not started with FREETOKEN_KV_RESTORE_VERIFY=1)")

    print("=== RESULT:", "PASS" if not FAILURES else "FAIL", "===")
    sys.exit(0 if not FAILURES else 1)

if __name__ == "__main__":
    main()

