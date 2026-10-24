import json
import torch
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    r"C:\AI_Models\Qwen3.8-Flash-Next-NVFP4", trust_remote_code=True)

tmp = r"C:\Users\X303\AppData\Local\Temp\opencode"
c = json.load(open(rf"{tmp}\long.json", encoding="utf-8"))["messages"][0]["content"]


def templated(content):
    msgs = [{"role": "user", "content": content}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(text, add_special_tokens=False)


ent = torch.load(
    r"C:\Users\X303\Documents\FreetokenP\kv_persist\c3bdce3c12cf5ef8\entries\e9755647afc2f114b5e0_0_22272.pt",
    map_location="cpu", weights_only=False)["tokens"]

cases = {
    "ALPHA": "SESSION-ALPHA-20261022. " + c,
    "UNDO": "SESSION-UNDO-20261022. " + c,
    "EMPTYTREE": "EMPTYTREE-123-I. " + c,
}
for name, content in cases.items():
    ids = torch.tensor(templated(content)[: ent.numel()], dtype=torch.int64)
    a = ids[:16].tolist()
    n = min(ids.numel(), ent.numel())
    eq = bool(torch.equal(ids[:n], ent[:n]))
    first_diff = next((i for i in range(n) if ids[i] != ent[i]), None)
    print(f"{name:10} tokens[:16]={a}")
    print(f"{'':10} n={n} equal_first_{n}={eq} first_diff_at={first_diff}")
print("entry  tokens[:16]=", ent[:16].tolist())
