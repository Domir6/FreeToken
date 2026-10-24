import io
import re
import sys

pages = sys.argv[1] if len(sys.argv) > 1 else "1610"
mamba = sys.argv[2] if len(sys.argv) > 2 else "4"

p = r"C:\Users\X303\Documents\FreetokenP\kv-rebuild-after-ready.ps1"
t = io.open(p, encoding="utf-8", newline="").read()
new_body = '{"moe_cache_size":512,"num_pages":%s,"num_mamba_slots":%s}' % (pages, mamba)
pat = re.compile(r'\{"moe_cache_size":512,"num_pages":\d+,"num_mamba_slots":\d+\}')
assert pat.search(t), "geometry body not found"
t = pat.sub(new_body, t, count=1)
io.open(p, "w", encoding="utf-8", newline="").write(t)
for line in t.splitlines():
    if "moe_cache_size" in line and "num_pages" in line:
        print("watcher:", line.strip())
print("budget check: pages*1622016 + mamba*115642376 =",
      int(pages) * 1622016 + int(mamba) * 115642376, "<= 3177893888")
