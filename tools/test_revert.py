"""Revert-then-resend simulation (the user's real workflow: see error -> revert -> resend).

Grow a session 3 turns, then "revert" to turn-1 length and resend with a real decode.
Before the PREFIX-hit fix this produced a full prefill (snapshot longer than request).
Run:  python tools/test_revert.py
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:1919"


def make_text(n_words, salt):
    return " ".join("%s%05x" % (salt, (i * 2654435761) & 0xFFFFF) for i in range(n_words))


def chat(text, max_tokens, timeout=1200):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    t0 = time.monotonic()
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    return resp, time.monotonic() - t0


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "rv"
    head = make_text(7940, tag + "head")          # ~60k
    turns = [make_text(260, f"{tag}t{i}") for i in range(3)]
    tail = ""
    for i, t in enumerate(turns):
        tail = (tail + " " + t).strip()
        resp, dur = chat((head + " " + tail).strip(), 300)
        pt = resp["usage"]["prompt_tokens"]
        print(f"  grow step{i + 1}: {dur:.2f}s ({pt} tok, decode 300) -> "
              f"{300 / dur:.0f} tok/s @ {pt} ctx", flush=True)
    # revert: back to turn-1 length, resend with real decode
    reverted = (head + " " + turns[0]).strip()
    resp, dur = chat(reverted, 500)
    pt = resp["usage"]["prompt_tokens"]
    ct = resp["usage"]["completion_tokens"]
    print(f"  REVERT resend: {dur:.2f}s ({pt} tok, decode {ct}) -> "
          f"{ct / dur:.0f} tok/s @ {pt} ctx", flush=True)
    print(f"  => {'PASS (prefix-view restore worked)' if dur < 10 else 'FAIL (full prefill?)'} "
          f"-- kv_boot.log should carry 'restore PREFIX hit'", flush=True)


if __name__ == "__main__":
    main()
