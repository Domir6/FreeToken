param(
    [int]$NumTokens = 98304,
    [int]$Port = 1919,
    [string]$LogDir = "C:\AI_Models\new_log",
    [string]$KvPersistDir = "C:\Users\X303\Documents\FreetokenP\kv_persist",
    [string]$Gpu = "GPU-54e316a5-5b97-c3dd-ebbc-d54e2f80cf22",
    [switch]$Detach,
    [switch]$NoMoeOverlap,
    [switch]$NoPrefetch,
    [switch]$PrefetchMeasure,
    [switch]$NoPersist,
    [switch]$NoSwap
)

$ISOLATED = "C:\Users\X303\Documents\FreetokenP"
$FT       = "C:\Users\X303\AppData\Local\FreeToken\venv\Scripts\ft.exe"
$MODEL    = "C:\AI_Models\Qwen3.8-Flash-Next-NVFP4"

if (-not (Test-Path -LiteralPath $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}
if (-not (Test-Path -LiteralPath $KvPersistDir)) {
    New-Item -ItemType Directory -Path $KvPersistDir -Force | Out-Null
}

$stamp   = Get-Date -Format "yyyyMMdd-HHmmss"
$logFile = if ($Detach) { Join-Path $LogDir "serve_isolated.log" } else { Join-Path $LogDir "freetoken-serve-$stamp.log" }
$errFile = "$logFile.err"

$env:PYTHONPATH = "$ISOLATED;$env:PYTHONPATH"
$env:TQDM_DISABLE = "1"
$env:FREETOKEN_KV_SWAP = if ($NoSwap) { "0" } else { "1" }
$env:FREETOKEN_KV_SWAP_HOST_BYTES = "17179869184"
$env:FREETOKEN_KV_SWAP_MIN_TOKENS = "256"
$env:FREETOKEN_KV_SWAP_PIN = "0"
$env:FREETOKEN_KV_SWAP_COPY_CHUNK = "2048"
$env:FREETOKEN_KV_PERSIST = if ($NoPersist -or $NoSwap) { "0" } else { "1" }
$env:FREETOKEN_KV_PERSIST_DIR = $KvPersistDir
$env:FREETOKEN_KV_PERSIST_BYTES = "68719476736"
$env:FREETOKEN_MOE_COPY_OVERLAP = if ($NoMoeOverlap) { "0" } else { "1" }
# measure costs ~25% decode throughput (it re-runs the gate on a side stream every step);
# keep it off by default and enable only for diagnostics.
$env:FREETOKEN_MOE_PREFETCH = if ($PrefetchMeasure -and -not $NoPrefetch) { "measure" } else { "off" }

$ftArgs = @(
    "serve",
    "--model", $MODEL,
    "--port", "$Port",
    "--moe-strategy", "offload",
    "--image-max-tokens", "512",
    "--memory-ratio", "0.95",
    "--num-tokens", "$NumTokens",
    "--moe-cache-size", "512",
    "--moe-cpu-layers", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,34,35,36,37,38,39,40,41,42,43,44,45,46,47",
    "--max-running-requests", "2",
    "--cuda-graph-max-bs", "1",
    "--moe-cpu-threads", "16",
    "--gpu", $Gpu,
    "--host", "127.0.0.1"
)

if ($Detach) {
    $proc = Start-Process -FilePath $FT -ArgumentList $ftArgs `
        -WorkingDirectory $ISOLATED -RedirectStandardOutput $logFile `
        -RedirectStandardError $errFile -PassThru -WindowStyle Hidden
    try { $proc.PriorityClass = "High" } catch { }
    Write-Host "Started detached FreeToken, PID=$($proc.Id), port=$Port, num_tokens=$NumTokens"
    Write-Host "stdout: $logFile"
    Write-Host "stderr: $errFile"
    Write-Host "kv persist: $KvPersistDir"
}
else {
    try {
        [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::High
        Write-Host " Priority: High" -ForegroundColor Green
    }
    catch { }

    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor DarkCyan
    Write-Host " FreeToken Qwen3.8-Flash-Next-NVFP4 (isolated, port $Port)" -ForegroundColor Cyan
    Write-Host " Log file: $logFile" -ForegroundColor Cyan
    Write-Host " KV persist dir: $KvPersistDir" -ForegroundColor Cyan
    Write-Host " Stop server with Ctrl+C" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor DarkCyan
    Write-Host ""

    & $FT @ftArgs 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) {
            $line = $_.Exception.Message
        }
        else {
            $line = "$_"
        }
        $line | Out-Host
        Add-Content -LiteralPath $logFile -Value $line -Encoding UTF8
    }
}
