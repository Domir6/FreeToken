import sys

path = r"C:\Users\X303\Documents\FreetokenP\start-freetoken-flash-next.ps1"
diag, verify = sys.argv[1], sys.argv[2]
raw = open(path, "rb").read()
assert raw[:3] == b"\xef\xbb\xbf", "BOM missing"
text = raw.decode("utf-8-sig")
before = text
text = text.replace('$env:FREETOKEN_KV_DIAG = "0"', f'$env:FREETOKEN_KV_DIAG = "{diag}"')
text = text.replace('$env:FREETOKEN_KV_DIAG = "1"', f'$env:FREETOKEN_KV_DIAG = "{diag}"')
text = text.replace('$env:FREETOKEN_KV_RESTORE_VERIFY = "0"', f'$env:FREETOKEN_KV_RESTORE_VERIFY = "{verify}"')
text = text.replace('$env:FREETOKEN_KV_RESTORE_VERIFY = "1"', f'$env:FREETOKEN_KV_RESTORE_VERIFY = "{verify}"')
open(path, "wb").write(b"\xef\xbb\xbf" + text.encode("utf-8"))
for line in text.splitlines():
    if "FREETOKEN_KV_DIAG" in line or "FREETOKEN_KV_RESTORE_VERIFY" in line:
        print("NOW:", line)
print("changed" if text != before else "unchanged")
