import json

tmp = r"C:\Users\X303\AppData\Local\Temp\opencode"
base = json.load(open(rf"{tmp}\long.json", encoding="utf-8"))
c = base["messages"][0]["content"]
H = "SESSION-UNDO-20261022. "


def write(name, content, mt=8):
    body = {"model": "Qwen3.8-Flash-Next-NVFP4",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": mt, "temperature": 0}
    json.dump(body, open(rf"{tmp}\{name}", "w", encoding="utf-8"), ensure_ascii=False)
    print(name, "chars", len(content))


write("u1.json", H + c)
write("u2.json", H + c[:81500])
write("ux.json", H + c)
write("compact.json",
      "COMPACTED-SUMMARY-V1. Key points: first, second, third. Continuation follows: " + c[-20000:])
print("done")
