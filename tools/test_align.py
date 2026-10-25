"""GDN/QSA alignment test: does a restored session decode IDENTICALLY to a radix-hot one?

GDN (linear attention) is a deterministic recurrence over the prefix, so for the same
prompt at temperature=0, "restored from host snapshot" and "radix-hot" must produce
byte-identical outputs. A mismatch = the snapshot's GDN state is positioned at a
different token than the KV restore point (the alignment bug class).
Run:  python tools/test_align.py
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:1919"


def make_text(n_words, salt):
    return " ".join("%s%05x" % (salt, (i * 2654435761) & 0xFFFFF) for i in range(n_words))


def chat(text, max_tokens, timeout=1200, temperature=0):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    t0 = time.monotonic()
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    return resp, time.monotonic() - t0


def out_text(resp):
    try:
        m = resp["choices"][0]["message"]
        r = m.get("reasoning_content") or m.get("reasoning") or ""
        return (r or "") + "||" + (m.get("content") or "")
    except Exception:
        return json.dumps(resp)[:200]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "al"
    head = make_text(7940, tag + "head")          # ~60k
    t1 = make_text(260, tag + "t1")
    t2 = make_text(260, tag + "t2")
    p1 = (head + " " + t1).strip()
    p2 = (head + " " + t1 + " " + t2).strip()

    print("  step A1: grow (decode 200) ...", flush=True)
    resp_a1, d = chat(p1, 200)
    ct1 = resp_a1["usage"]["completion_tokens"]
    print(f"    {d:.1f}s ct={ct1} out[:60]={out_text(resp_a1)[:60]!r}", flush=True)
    time.sleep(8)

    print("  step A2: extend (decode 200) -- this one goes through RESTORE ...", flush=True)
    resp_a2, d = chat(p2, 200)
    a2 = out_text(resp_a2)
    ct2 = resp_a2["usage"]["completion_tokens"]
    print(f"    {d:.1f}s ct={ct2} out[:60]={a2[:60]!r}", flush=True)
    time.sleep(2)

    print("  step B: same prompt again (radix-hot, decode 200) ...", flush=True)
    resp_b, d = chat(p2, 200)
    b = out_text(resp_b)
    ctb = resp_b["usage"]["completion_tokens"]
    print(f"    {d:.1f}s ct={ctb} out[:60]={b[:60]!r}", flush=True)

    same = a2 == b and ct2 == ctb
    print(f"  => outputs {'IDENTICAL' if same else 'DIVERGE'} "
          f"({'GDN aligned' if same else 'GDN MISALIGNED -- correctness bug'}) "
          f"(content_len {len(a2)}/{len(b)}, completion {ct2}/{ctb})", flush=True)
    # full dumps for tripartite arbitration (A2 vs B vs clean-prefill C)
    import io as _io
    for nm, txt in (("a2", a2), ("b", b)):
        _io.open(rf"C:\Users\X303\AppData\Local\Temp\opencode\align_{nm}.txt", "w",
                 encoding="utf-8").write(txt)
    if not same:
        # first differing char
        for i, (x, y) in enumerate(zip(a2, b)):
            if x != y:
                print(f"     first diff at char {i}: {a2[max(0,i-30):i+30]!r} vs {b[max(0,i-30):i+30]!r}", flush=True)
                break
        else:
            print(f"     length diff: {len(a2)} vs {len(b)}", flush=True)
    print("=== RESULT:", "PASS" if same else "FAIL", "===")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
