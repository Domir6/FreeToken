import json

tmp = r"C:\Users\X303\AppData\Local\Temp\opencode"
base = json.load(open(rf"{tmp}\long.json", encoding="utf-8"))
c = base["messages"][0]["content"]


def write(name, content, max_tokens=1):
    body = {"model": "Qwen3.8-Flash-Next-NVFP4",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0}
    with open(rf"{tmp}\{name}", "w", encoding="utf-8") as fh:
        json.dump(body, fh, ensure_ascii=False)
    print(name, "content_len", len(content))


write("t2a.json", "SESSION-ALPHA-20261022. " + c + "\n\nNow list exactly three key points from the text above, numbered.")
write("t2b.json", "SESSION-ALPHA-20261022. " + c + "\n\nNow write one short sentence summarizing the text above.")
write("t3r2.json", "SESSION-ALPHA-20261022. " + c[:30000] + "\n\nSecond turn: what is the first key point?")
write("short.json", "hello, what is 2+2?")
print("done")
