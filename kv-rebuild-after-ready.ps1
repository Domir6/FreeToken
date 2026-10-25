param([int]$Port = 1919, [string]$LogDir = "C:\AI_Models\logs")

# Wait until the server reports status=ok, then rebuild the cache geometry:
#   MoE expert cache 512 slots, KV 1648 pages (103040 tokens), mamba 4 slots.
# Booting with the auto geometry (1234 slots, prefill overlap ON) is what makes the
# engine allocate the two full expert-layer buffers; rebuilding down to 512 keeps them,
# so prefill keeps its overlapped expert copies. Starting directly with
# "--moe-cache-size 512 --disable-moe-prefill-overlap" only allocates one buffer and
# makes prefill ~5x slower.
$body = '{"moe_cache_size":512,"num_pages":1360,"num_mamba_slots":4}'
$log = Join-Path $LogDir "kv_rebuild.log"
$deadline = (Get-Date).AddMinutes(15)

while ((Get-Date) -lt $deadline) {
    try {
        $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5
        if ($h.status -eq "ok") {
            for ($attempt = 1; $attempt -le 12; $attempt++) {
                try {
                    $r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/cache/rebuild" -Method Post `
                        -ContentType "application/json" -Body $body -TimeoutSec 600
                    ("{0} rebuild status={1} moe={2} pages={3} mamba={4} error={5} attempt={6}" -f `
                        (Get-Date -Format "HH:mm:ss"), $r.status, $r.moe_cache_size, $r.num_pages, $r.mamba_slots, $r.error, $attempt) |
                        Out-File $log -Append -Encoding utf8
                    break
                } catch {
                    ("{0} rebuild attempt {1} failed: {2}" -f (Get-Date -Format "HH:mm:ss"), $attempt, $_.Exception.Message) |
                        Out-File $log -Append -Encoding utf8
                    Start-Sleep -Seconds 5
                }
            }
            break
        }
    } catch { }
    Start-Sleep -Seconds 5
}
