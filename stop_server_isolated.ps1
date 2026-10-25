param(
    [int]$Port = 1919,
    [int]$TimeoutSec = 60,
    [string]$KvPersistDir = "C:\Users\X303\Documents\FreetokenP\kv_persist",
    [switch]$Force
)

# Stop the isolated FreeToken instance on $Port.
#
# Default: try CTRL_C_EVENT first (orderly uvicorn shutdown), then fall back to
# terminating the process tree. -Force skips the CTRL_C attempt.
#
# Host KV is flushed to SSD at every idle transition, so killing the workers does not
# lose prefixes captured by completed requests.
#
# Orphan sweep: Windows spawn workers whose parent died keep the whole model in VRAM.
# After the listener is gone, any `python -c spawn_main` process whose parent_pid is no
# longer alive belongs to a dead server and is terminated too.

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class ConsoleCtl {
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool FreeConsole();
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool AttachConsole(uint dwProcessId);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool GenerateConsoleCtrlEvent(uint dwCtrlEvent, uint dwProcessGroupId);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool SetConsoleCtrlHandler(IntPtr handler, bool add);
}
"@ -ErrorAction SilentlyContinue

function Get-ProcessTree {
    param([int]$RootPid)
    $all = Get-CimInstance Win32_Process
    $ids = New-Object 'System.Collections.Generic.HashSet[int]'
    [void]$ids.Add($RootPid)
    $frontier = @($RootPid)
    while ($frontier.Count -gt 0) {
        $next = @()
        foreach ($p in $frontier) {
            foreach ($c in $all) {
                if ([int]$c.ParentProcessId -eq $p -and $ids.Add([int]$c.ProcessId)) {
                    $next += [int]$c.ProcessId
                }
            }
        }
        $frontier = $next
    }
    return @($ids)
}

function Stop-Tree {
    param([int]$RootPid)
    $tree = Get-ProcessTree -RootPid $RootPid
    foreach ($procId in ($tree | Sort-Object -Descending)) {
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
    return $tree.Count
}

function Stop-OrphanWorkers {
    $all = Get-CimInstance Win32_Process
    $alive = @{}
    foreach ($p in $all) { $alive[[int]$p.ProcessId] = $true }
    $orphans = @($all | Where-Object {
        $_.Name -eq 'python.exe' -and
        $_.CommandLine -match 'spawn_main' -and
        -not $alive.ContainsKey([int]$_.ParentProcessId)
    })
    foreach ($o in $orphans) {
        Stop-Process -Id $o.ProcessId -Force -ErrorAction SilentlyContinue
    }
    return $orphans.Count
}

function Wait-PortFree {
    param([int]$Port, [int]$Seconds)
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)) {
            return $true
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

$conn = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if (-not $conn) {
    Write-Output "Nothing is listening on port $Port."
}
else {
    $root = [int]($conn.OwningProcess | Select-Object -First 1)

    if (-not $Force) {
        # The API-server main process can be a frozen bundle that never sees our patched
        # routes, so trigger the dump through a flag file the (patched) scheduler consumes on
        # its next idle transition. A 1-token request produces that transition.
        $flag = Join-Path $KvPersistDir ".dump_request"
        New-Item -ItemType Directory -Path $KvPersistDir -Force | Out-Null
        Set-Content -LiteralPath $flag -Value "dump" -Encoding ASCII
        Write-Output "Requesting KV dump (GPU prefixes -> host -> SSD)..."
        try {
            $body = '{"model":"Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"ping"}],"max_tokens":1}'
            Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
                -ContentType "application/json" -Body $body -TimeoutSec 120 | Out-Null
        }
        catch {
            Write-Output ("Dump trigger request failed: " + $_.Exception.Message)
        }
        $deadline = (Get-Date).AddSeconds(300)
        while ((Test-Path -LiteralPath $flag) -and ((Get-Date) -lt $deadline)) {
            Start-Sleep -Seconds 2
        }
        if (Test-Path -LiteralPath $flag) {
            Write-Output "KV dump flag was not consumed within 300s; proceeding with stop."
        }
        else {
            Write-Output "KV dump completed."
        }
    }

    if ($Force) {
        $n = Stop-Tree -RootPid $root
        Write-Output "Force-stopped $n process(es) in the tree of PID $root."
    }
    else {
        Write-Output "Sending CTRL_C_EVENT to PID $root (port $Port)..."
        [void][ConsoleCtl]::FreeConsole()
        if ([ConsoleCtl]::AttachConsole([uint32]$root)) {
            [void][ConsoleCtl]::SetConsoleCtrlHandler([IntPtr]::Zero, $true)
            [void][ConsoleCtl]::GenerateConsoleCtrlEvent(0, 0)
            [void][ConsoleCtl]::FreeConsole()
        }
        else {
            Write-Output "Could not attach to the target console; falling back to terminate."
        }

        if (Wait-PortFree -Port $Port -Seconds $TimeoutSec) {
            Write-Output "Port $Port released after graceful shutdown."
        }
        else {
            # CTRL_C can stall forever when the core is inside a long C call (big gather /
            # SSD write during its own shutdown dump). CTRL_BREAK cannot be ignored by
            # Python on Windows: the process terminates itself cleanly (atexit handlers
            # still run), which is a step above tree-kill. The KV dump has already been
            # taken via the flag file, so no data is at risk at this point.
            Write-Output "Graceful stop timed out; sending CTRL_BREAK_EVENT..."
            [void][ConsoleCtl]::FreeConsole()
            if ([ConsoleCtl]::AttachConsole([uint32]$root)) {
                [void][ConsoleCtl]::SetConsoleCtrlHandler([IntPtr]::Zero, $true)
                [void][ConsoleCtl]::GenerateConsoleCtrlEvent(1, 0)
                [void][ConsoleCtl]::FreeConsole()
            }
            if (Wait-PortFree -Port $Port -Seconds 15) {
                Write-Output "Port $Port released after CTRL_BREAK."
            }
            else {
                Write-Output "CTRL_BREAK also timed out; terminating the process tree."
                [void](Stop-Tree -RootPid $root)
            }
        }
    }
}

    $totalReaped = 0
    for ($i = 0; $i -lt 3; $i++) {
        if ($i -gt 0) { Start-Sleep -Seconds 2 }
        $totalReaped += Stop-OrphanWorkers
    }
    if ($totalReaped -gt 0) {
        Write-Output "Reaped $totalReaped orphaned backend worker(s) (parent already gone)."
    }

Start-Sleep -Seconds 2
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    Write-Output "Port $Port is still in use."
    exit 1
}
Write-Output "Port $Port released."
