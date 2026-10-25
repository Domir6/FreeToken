"""ACC1 rebuild (spec v1.1 ruling 8): 8 independent ~22k-token sessions, two rotation laps.

Acceptance criteria are hard-coded and printed as PASS/FAIL lines:
  [ACC-1] lap-2 revisit of a session whose snapshot left the GPU (8 sessions > 5 mamba
          slots and 8x22k tokens > 92160-token KV pool) completes WITHOUT a full prefill
          (client-observed duration < FULL_PREFILL_BUDGET_S)
  [ACC-2] the host tier reports restored-count growth during lap 2 (host restore path used)
  [ACC-3] with FREETOKEN_KV_RESTORE_VERIFY=1 every new restore_verify.log line shows max|d|=0
  [ACC-4] idle gap = 0: after an idle wait the staging ledger is flat
          (staged == drained + inflight, i.e. no backlog)

Run while the server is up:  python tools/test_rotation.py
"""
import io
import json
import re
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:1919"
PRELOAD_LOG = r"C:\AI_Models\logs\kv_preload.log"
VERIFY_LOG = r"C:\AI_Models\logs\restore_verify.log"
N_SESSIONS = 8
TARGET_TOKENS = 22_000
FULL_PREFILL_BUDGET_S = 7.0  # 22k @ 1783 t/s ~ 12.3 s; a hit must land well under this
FAILURES = []


def check(cond, msg, extra=""):
    print(("PASS " if cond else "FAIL ") + msg + (("  <- " + extra) if extra and not cond else ""))
    if not cond:
        FAILURES.append(msg)


def chat(text, timeout=600):
    body = json.dumps(
        {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": 1}
    ).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    t0 = time.monotonic()
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    return resp, time.monotonic() - t0


def make_text(n_words, salt):
    words = []
    for i in range(n_words):
        words.append("%s%05x" % (salt, (i * 2654435761) & 0xFFFFF))
    return " ".join(words)


def tail(path):
    try:
        return io.open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return []


def summary_field(line, name):
    m = re.search(name + r"=(\d+)", line)
    return int(m.group(1)) if m else None


def main():
    sessions = []
    for i in range(N_SESSIONS):
        sessions.append(make_text(7000, "s%xz" % i))

    print(f"warm-up: calibrating prompt length with session 0 ...")
    resp, _ = chat(sessions[0])
    got = resp["usage"]["prompt_tokens"]
    scale = TARGET_TOKENS / max(got, 1)
    if abs(scale - 1.0) > 0.05:
        n_words = max(100, int(7000 * scale))
        for i in range(N_SESSIONS):
            sessions[i] = make_text(n_words, "s%xz" % i)
        resp, _ = chat(sessions[0])
        got = resp["usage"]["prompt_tokens"]
    print(f"calibrated: prompt_tokens ~= {got}")

    verify_before = len(tail(VERIFY_LOG))

    print(f"lap 1: {N_SESSIONS} sessions (cold-ish, builds the cache) ...")
    lap1 = []
    for i, text in enumerate(sessions):
        _, dur = chat(text)
        lap1.append(dur)
        print(f"  s{i + 1}: {dur:.1f}s")

    print("waiting for idle drain ...")
    time.sleep(12)

    preload_before = tail(PRELOAD_LOG)
    print(f"lap 2: {N_SESSIONS} revisits (snapshots left GPU: {N_SESSIONS} > mamba slots) ...")
    lap2 = []
    for i, text in enumerate(sessions):
        _, dur = chat(text)
        lap2.append(dur)
        print(f"  s{i + 1}: {dur:.1f}s")

    # [ACC-4] idle gap = 0: give the idle gate a beat, then read the staging ledger
    print("waiting for idle gate to flush the staging ledger ...")
    time.sleep(35)
    lines = tail(PRELOAD_LOG)
    new_lines = lines[len(preload_before):] if len(lines) > len(preload_before) else []
    ledger_ok = None
    for line in reversed(new_lines):
        st = summary_field(line, "staged")
        dr = summary_field(line, "drained")
        infl = summary_field(line, "inflight")
        if st is not None and dr is not None and infl is not None:
            ledger_ok = st == dr + infl
            print(f"  staging ledger: staged={st} drained={dr} inflight={infl}")
            break
    if ledger_ok is None:
        new_all = lines[len([l for l in preload_before]) :]
        check(False, "[ACC-4] staging ledger line seen after lap 2", "no mamba_staging in new kv_preload rows")
    else:
        check(ledger_ok, "[ACC-4] idle gap = 0: staging ledger flat (staged == drained + inflight)")

    # [ACC-1] lap 2 revisits must not be full prefills
    worst = max(lap2)
    check(worst < FULL_PREFILL_BUDGET_S,
          f"[ACC-1] every lap-2 revisit below {FULL_PREFILL_BUDGET_S}s (worst {worst:.1f}s)")

    # [ACC-2] host restore actually ran during lap 2
    restored_before = summary_field(preload_before[-1], "restored") if preload_before else 0
    restored_after = summary_field(lines[-1], "restored") if lines else 0
    check(restored_after > restored_before,
          f"[ACC-2] host restore fired during lap 2 (restored {restored_before} -> {restored_after})")

    # [ACC-3] numeric parity of restores (only meaningful with RESTORE_VERIFY=1)
    new_verify = tail(VERIFY_LOG)[verify_before:]
    kv_lines = [l for l in new_verify if l.startswith("KV restore verify")]
    mb_lines = [l for l in new_verify if l.startswith("mamba restore verify")]
    if kv_lines or mb_lines:
        bad = [l for l in kv_lines if "max|dK|=0" not in l or "max|dV|=0" not in l]
        bad += [l for l in mb_lines if "max|d|=0" not in l]
        check(not bad, f"[ACC-3] {len(kv_lines)} KV + {len(mb_lines)} mamba verify lines all max|d|=0",
              "; ".join(bad[:2]))
    else:
        print("INFO [ACC-3] no restore_verify lines (server started without FREETOKEN_KV_RESTORE_VERIFY=1)")

    print("=== RESULT:", "PASS" if not FAILURES else "FAIL", "===")
    sys.exit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
