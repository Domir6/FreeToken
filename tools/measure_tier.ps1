param([string]$Tag = "T1")

$tmp = "C:\Users\X303\AppData\Local\Temp\opencode"
$j = Get-Content "$tmp\long.json" -Raw | ConvertFrom-Json
$c = $j.messages[0].content
$nonce = "$Tag-" + (Get-Random) + ". "
$j.messages[0].content = $nonce + $c + $c + $c
$j.max_tokens = 1
$json = $j | ConvertTo-Json -Depth 6 -Compress
$sw = [Diagnostics.Stopwatch]::new()
$sw.Restart()
$r = Invoke-RestMethod -Uri "http://127.0.0.1:1919/v1/chat/completions" -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ([Text.Encoding]::UTF8.GetBytes($json)) -TimeoutSec 1800
$sw.Stop()
$pt = $r.usage.prompt_tokens
"[{0}] PREFILL: prompt={1} elapsed={2}s => {3} t/s" -f $Tag, $pt, [math]::Round($sw.Elapsed.TotalSeconds, 2), [math]::Round($pt / $sw.Elapsed.TotalSeconds, 0)

$body = '{"model":"Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"Write a long detailed technical essay about attention mechanisms in transformers."}],"max_tokens":1000,"temperature":0}'
$sw.Restart()
$r2 = Invoke-RestMethod -Uri "http://127.0.0.1:1919/v1/chat/completions" -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 900
$sw.Stop()
$ct = $r2.usage.completion_tokens
"[{0}] DECODE : completion={1} elapsed={2}s => {3} t/s" -f $Tag, $ct, [math]::Round($sw.Elapsed.TotalSeconds, 2), [math]::Round(($ct - 1) / $sw.Elapsed.TotalSeconds, 1)

$os = Get-CimInstance Win32_OperatingSystem
$w = Get-Process | Where-Object { $_.ProcessName -eq 'python' } | Sort-Object WorkingSet64 -Descending | Select-Object -First 1
"[{0}] MEM    : python WS={1}GiB freeRAM={2}GiB" -f $Tag, [math]::Round($w.WorkingSet64 / 1GB, 1), [math]::Round($os.FreePhysicalMemory / 1MB, 1)
$s = Invoke-RestMethod "http://127.0.0.1:1919/v1/cache/status" -TimeoutSec 10
"[{0}] GEOM   : pages={1} moe={2} mamba={3}" -f $Tag, $s.geometry.num_pages, $s.geometry.moe_cache_size, $s.geometry.num_mamba_slots
