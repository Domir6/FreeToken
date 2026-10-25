# ==========================================================
# FreeToken Qwen3.8-Flash-Next-NVFP4 启动脚本
# 由 start-freetoken-flash-next.bat 调用，请勿直接运行
# ==========================================================
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$OutputEncoding = [Text.Encoding]::UTF8

# 提升本进程优先级：ft.exe 及其派生的 python core 都会继承 High
try {
    [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::High
    Write-Host " Priority: High (inherited by ft.exe and python core)" -ForegroundColor Green
} catch {
    Write-Host (" Priority set failed: " + $_.Exception.Message) -ForegroundColor Yellow
}

# tqdm 保持启用：输出不经管道/重定向，stderr 直连控制台，进度条原地刷新到 100%
$env:TQDM_MININTERVAL = "0.1"

# ---- 访问令牌（如需把 1919 暴露到公网/局域网）：从同目录 .api_key 读取 ----
# FreeToken 会读 $env:FREETOKEN_API_KEY 并要求除 /health 外的所有路由带 Bearer/x-api-key。
# 文件不存在则不开启认证。.api_key 已被 .gitignore 忽略，不要提交。
$apiKeyFile = Join-Path $PSScriptRoot ".api_key"
if (Test-Path -LiteralPath $apiKeyFile) {
    $env:FREETOKEN_API_KEY = (Get-Content -LiteralPath $apiKeyFile -Raw).Trim()
    Write-Host " API key: loaded from .api_key (auth ENABLED)" -ForegroundColor Green
}

# ---- 隔离补丁层 + KV 三层缓存 ----
# 让 freetoken 的纯 Python 模块优先从本目录加载（scheduler / layers / moe / kv_swap ...）
$env:PYTHONPATH = "C:\Users\X303\Documents\FreetokenP;$env:PYTHONPATH"
# GPU KV -> pinned host 内存 -> SSD
# KV swap + SSD 持久化默认开启；如需关闭，取消下一行注释并设为 "0"
# $env:FREETOKEN_KV_SWAP = "0"
$env:FREETOKEN_KV_SWAP_HOST_BYTES = "8589934592"
$env:FREETOKEN_KV_SWAP_MIN_TOKENS = "256"
$env:FREETOKEN_KV_SWAP_PIN = "0"
$env:FREETOKEN_KV_SWAP_COPY_CHUNK = "2048"
# 驱逐前 mamba 快照落盘（pinned staging 异步版）：默认开；设 0 回退到“丢弃”行为
$env:FREETOKEN_KV_SWAP_MAMBA_ON_EVICT = "1"
# SSD 持久化（每个请求结束的 idle 立即落盘）
# SSD 落盘默认开启；关闭：$env:FREETOKEN_KV_PERSIST = "0"
$env:FREETOKEN_KV_PERSIST_DIR = "C:\Users\X303\Documents\FreetokenP\kv_persist"
$env:FREETOKEN_KV_PERSIST_BYTES = "68719476736"
# 暂时关闭 SSD 恢复（restore 路径有页归属问题，待修复）
# SSD 恢复默认开启；关闭：$env:FREETOKEN_KV_RESTORE = "0"
$env:FREETOKEN_KV_RESTORE_VERIFY = "0"
# MoE 侧流拷贝（免费提速）；measure 会掉 ~25% 吞吐，生产保持 off
$env:FREETOKEN_MOE_COPY_OVERLAP = "1"
$env:FREETOKEN_MOE_PREFETCH = "off"

$logDir  = "C:\AI_Models\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = Join-Path $logDir ("freetoken-serve-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".log")

Write-Host ""
Write-Host "==========================================================" -ForegroundColor DarkCyan
Write-Host " FreeToken Qwen3.8-Flash-Next-NVFP4  (port 1919)" -ForegroundColor Cyan
Write-Host (" Log file: " + $logFile) -ForegroundColor Cyan
Write-Host " Stop server: run scripts\stop_server_isolated.ps1 (saves KV)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor DarkCyan
Write-Host ""

$ft = "C:\Users\X303\AppData\Local\FreeToken\venv\Scripts\ft.exe"

$arguments = @(
    "serve",
    "--model", "C:\AI_Models\Qwen3.8-Flash-Next-NVFP4",
    "--port", "1919",
    "--moe-strategy", "auto",
    "--moe-cpu-layers", "auto",
    "--text-model-only",
    "--memory-ratio", "0.92",
    "--num-tokens", "87040",
    "--max-running-requests", "1",
    "--moe-cpu-threads", "32",
    "--host", "127.0.0.1",
    "--gpu", "GPU-54e316a5-5b97-c3dd-ebbc-d54e2f80cf22",
    "--cors-origins", "tauri://localhost,http://tauri.localhost,http://localhost:1420"
)

# Run the exe directly (no pipe / no 2>&1): redirecting stderr makes it a pipe, which
# makes tqdm think it is not a terminal and print one line per update. A transcript keeps
# a log file without touching the child's console handles, so tqdm refreshes in place.
# Boot with the auto MoE geometry (1234 slots, prefill overlap ON) so the engine allocates
# the two full expert-layer buffers, then rebuild to 512 slots / 98304-token KV / 4 mamba
# slots once the server is ready. Starting directly with "--moe-cache-size 512
# --disable-moe-prefill-overlap" only allocates ONE buffer and makes prefill ~5x slower.
Start-Process powershell -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',
    (Join-Path $PSScriptRoot 'kv-rebuild-after-ready.ps1'), '-Port','1919') -WindowStyle Hidden

Start-Transcript -Path $logFile -Force | Out-Null
try {
    & $ft @arguments
}
finally {
    try { Stop-Transcript | Out-Null } catch { }
}
$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host (" Server exited (code $exitCode). Log saved: " + $logFile) -ForegroundColor Yellow
