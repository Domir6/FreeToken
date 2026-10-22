param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ISOLATED = "C:\Users\X303\Documents\FreetokenP"
$VENV_PY  = "C:\Users\X303\AppData\Local\FreeToken\venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VENV_PY)) {
    Write-Error "venv python not found: $VENV_PY"
    exit 1
}

$env:PYTHONPATH = "$ISOLATED;$env:PYTHONPATH"

& $VENV_PY -c @"
import sys, freetoken
print('freetoken loaded from:', freetoken.__file__)
print('isolated:', freetoken.__file__.startswith(r'$ISOLATED'))
print('version:', freetoken.__version__)
from freetoken.moe import offload_cache
print('offload_cache:', offload_cache.__file__)
print('is .py:', offload_cache.__file__.endswith('.py'))
"@

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Args.Count -gt 0) {
    & $VENV_PY @Args
}
