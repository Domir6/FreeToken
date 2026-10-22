param(
    [int]$IntervalMinutes = 30,
    [string]$Port = 1919,
    [string]$LogDir = "C:\AI_Models\logs",
    [int]$StopAfter = 0   # 0 = run forever; N = stop after N samples (for validation)
)

# uptime watcher: sample prefill throughput + free RAM + uptime on a fixed cadence so we can
# draw a restart threshold line. The load probes are deliberately small (~7k tokens, fresh
# random prefix => always a cold prefill) so they do not disturb real traffic much; whatever
# KV they leave behind gets evicted like any other completed request.
$csv = Join-Path $LogDir "uptime_watch.csv"
$probe = Join-Path $env:TEMP "uptime_probe_body.txt"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# build the probe body once (stable text; the prefix is randomised per sample to force a miss)
$longJson = "C:\Users\X303\AppData\Local\Temp\opencode\long.json"
if (Test-Path $longJson) {
    $obj = Get-Content $longJson -Raw | ConvertFrom-Json
    $body = $obj.messages[0].content.Substring(0, [Math]::Min(32000, $obj.messages[0].content.Length))
    Set-Content -LiteralPath $probe -Value $body -Encoding UTF8
} else {
    Set-Content -LiteralPath $probe -Value ("lorem ipsum dolor sit amet " * 1200) -Encoding UTF8
}
$bodyText = Get-Content -LiteralPath $probe -Raw

if (-not (Test-Path $csv)) {
    "timestamp,uptime_min,freeRAM_GiB,prefill_tps,gpu_MiB,status" | Out-File $csv -Encoding utf8
}

$n = 0
while ($true) {
    $n++
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $status = "ok"
    $uptimeMin = ""
    $freeRam = [Math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
    $gpu = ""
    try { $gpu = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader) } catch { $gpu = "n/a" }
    $tps = ""

    try {
        $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 10
        if ($h.status -ne "ok") { $status = "loading" }
        if ($h.uptime_s) { $uptimeMin = [Math]::Round($h.uptime_s / 60.0, 1) }
    } catch { $status = "DOWN" }

    if ($status -eq "ok") {
        $nonce = "WATCH-" + (Get-Random) + "-" + (Get-Date -Format "HHmmss") + ". "
        $payload = @{ model = "Qwen3.8-Flash-Next-NVFP4"
                      messages = @(@{ role = "user"; content = $nonce + $bodyText })
                      max_tokens = 1; temperature = 0 } | ConvertTo-Json -Depth 6 -Compress
        try {
            $sw = [Diagnostics.Stopwatch]::new()
            $sw.Restart()
            $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
                -ContentType "application/json; charset=utf-8" `
                -Body ([Text.Encoding]::UTF8.GetBytes($payload)) -TimeoutSec 600
            $sw.Stop()
            $pt = $r.usage.prompt_tokens
            $tps = [Math]::Round($pt / $sw.Elapsed.TotalSeconds, 0)
        } catch {
            $status = "probe_failed"
        }
    }

    $line = "{0},{1},{2},{3},{4},{5}" -f $ts, $uptimeMin, $freeRam, $tps, $gpu, $status
    $line | Out-File $csv -Append -Encoding utf8
    Write-Output $line

    if ($StopAfter -gt 0 -and $n -ge $StopAfter) { break }
    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
