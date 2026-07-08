param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = "D:\anaconda3\envs\x-anylabeling\python.exe"
$appPath = Join-Path $repoRoot "anylabeling\app.py"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python environment not found: $pythonExe"
}

if (-not (Test-Path -LiteralPath $appPath)) {
    throw "Application entrypoint not found: $appPath"
}

$env:PYTHONPATH = $repoRoot

Push-Location $repoRoot
try {
    if (-not $AppArgs -or $AppArgs.Count -eq 0) {
        & $pythonExe $appPath --no-auto-update-check
    } else {
        & $pythonExe $appPath @AppArgs
    }
} finally {
    Pop-Location
}
