"""Three-scenario dual-session acceptance (user protocol, 2026-10-25):
  1. two small (10k) sessions alternating   -> expect pure radix HITs (pool holds both)
  2. two large (75k) sessions alternating   -> pool forces eviction -> lap-2 must come
     back via the whole-session restore (<=3s); "compact" (eviction) is visible in the
     preload/eviction logs
  3. two sub-agents: 60k fixed head + 10 growing turns ("talk a lot"), interleaved ->
     step-wise hit rate + comeback restore
Run:  python tools/test_dual_switch.py <scenario 1|2|3>
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:1919"


def chat(text, timeout=900, max_tokens=1):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    t0 = time.monotonic()
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    return resp, time.monotonic() - t0


def make_text(n_words, salt):
    return " ".join("%s%05x" % (salt, (i * 2654435761) & 0xFFFFF) for i in range(n_words))


def calibrate(target, tag):
    body = json.dumps({"model": "x", "max_tokens": 1,
                       "messages": [{"role": "user", "content": make_text(1000, tag + "0")}]}).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    resp = json.load(urllib.request.urlopen(req, timeout=600))
    got = resp["usage"]["prompt_tokens"]
    n = int(1000 * target / got)
    print(f"  calibrated: {got} tok/1000w -> {n} words for {target}k", flush=True)
    return n


def shape_of(prompt_tokens, ttft_s, dur_s):
    if not prompt_tokens or not ttft_s:
        return "?"
    ratio = prompt_tokens / ttft_s
    return "HIT " if ratio > 8000 else ("PART" if ratio > 3000 else "MISS")


def run_pair(sessions, laps, label):
    print(f"  {label}")
    results = []
    for lap in range(laps):
        lap_durs = []
        for i, text in enumerate(sessions):
            resp, dur = chat(text)
            ttft = (resp["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
            pt = resp["usage"]["prompt_tokens"]
            lap_durs.append((dur, pt))
            print(f"    lap{lap + 1} s{i + 1}: {dur:.2f}s ({pt} tok) -> {dur / (pt / 1000):.0f} tok/s", flush=True)
        results.append(lap_durs)
        time.sleep(12)
    return results


def main():
    sc = sys.argv[1] if len(sys.argv) > 1 else "1"

    if sc == "1":
        n = calibrate(10_000, "sc1")
        sessions = [make_text(n, "sc1a"), make_text(n, "sc1b")]
        res = run_pair(sessions, 2, "scenario 1: two 10k sessions alternate")
        lap2 = [d for d, _ in res[1]]
        worst = max(lap2)
        print(f"  => lap2 worst {worst:.2f}s: {'PASS' if worst < 2.5 else 'FAIL'} "
              f"(pool holds both, pure radix expected)")

    elif sc == "2":
        n = calibrate(75_000, "sc2")
        sessions = [make_text(n, "sc2a"), make_text(n, "sc2b")]
        res = run_pair(sessions, 2, "scenario 2: two 75k sessions alternate (eviction/compact forced)")
        lap2 = [d for d, _ in res[1]]
        worst = max(lap2)
        print(f"  => lap2 worst {worst:.2f}s: {'PASS' if worst <= 3.0 else 'FAIL'} "
              f"(whole-session restore budget 3s; full prefill was {max(d for d, _ in res[0]):.0f}s)")

    elif sc == "3":
        head_n = calibrate(60_000, "sc3")
        heads = [make_text(head_n, "sc3a"), make_text(head_n, "sc3b")]
        print(f"  scenario 3: two sub-agents (60k head + 10 growing turns each), interleaved")
        turns_a, turns_b = [], []
        tails = ["", ""]
        for step in range(10):
            for i, (head, tail_acc) in enumerate(zip(heads, tails)):
                turn_txt = make_text(260, f"sc3t{i}s{step}")
                tails[i] = (tail_acc + " " + turn_txt).strip()
                resp, dur = chat((head + " " + tails[i]).strip(), max_tokens=300)
                pt = resp["usage"]["prompt_tokens"]
                ct = resp["usage"]["completion_tokens"]
                tag = "A" if i == 0 else "B"
                print(f"    agent{tag} step{step + 1}: {dur:.2f}s ({pt} tok, decode {ct}) "
                      f"-> total {ct / dur:.0f} tok/s @ {pt} ctx", flush=True)
                (turns_a if i == 0 else turns_b).append(dur)
            time.sleep(30)
        import statistics
        med_a = statistics.median(turns_a)
        med_b = statistics.median(turns_b)
        print(f"  => median step time A {med_a:.2f}s B {med_b:.2f}s "
              f"(60k head re-restore each step would be >=1.6s; pure radix <1.5s)")

    print(f"=== scenario {sc} done ===")


if __name__ == "__main__":
    main()
