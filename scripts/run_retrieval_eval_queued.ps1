# Queue GPU-heavy eval: exclusive lock DATA/eval_gpu.lock under repo root.
# Usage: .\scripts\run_retrieval_eval_queued.ps1 python -m eval.run_profile_metrics --profiles matrix
# Optional: $env:EVAL_QUEUE_LOCK_TIMEOUT_SEC = seconds to wait (default 86400). $env:EVAL_QUEUE_POLL_SEC = poll interval (default 3).

[CmdletBinding()]
param(
  [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
  [string[]] $CommandParts
)

$ErrorActionPreference = 'Stop'
$timeoutSec = 86400
if ($env:EVAL_QUEUE_LOCK_TIMEOUT_SEC -match '^\d+$') {
  $tmp = [int]$env:EVAL_QUEUE_LOCK_TIMEOUT_SEC
  if ($tmp -gt 0) { $timeoutSec = $tmp }
}
$pollSec = 3
if ($env:EVAL_QUEUE_POLL_SEC -match '^\d+$') {
  $tmp2 = [int]$env:EVAL_QUEUE_POLL_SEC
  if ($tmp2 -gt 0) { $pollSec = $tmp2 }
}

$root = Split-Path -Parent $PSScriptRoot
$lockPath = Join-Path $root 'DATA\eval_gpu.lock'
$dataDir = Split-Path -Parent $lockPath
if (-not (Test-Path $dataDir)) { New-Item -ItemType Directory -Path $dataDir | Out-Null }

$deadline = [datetime]::UtcNow.AddSeconds([math]::Max(1, $timeoutSec))
$fileStream = $null

while ($true) {
  try {
    $fileStream = [System.IO.File]::Open(
      $lockPath,
      [System.IO.FileMode]::OpenOrCreate,
      [System.IO.FileAccess]::ReadWrite,
      [System.IO.FileShare]::None
    )
    break
  } catch {
    if ([datetime]::UtcNow -ge $deadline) {
      throw "Timeout waiting for lock: $lockPath (${timeoutSec}s)"
    }
    Start-Sleep -Seconds $pollSec
  }
}

try {
  $exe = $CommandParts[0]
  $pass = @()
  if ($CommandParts.Count -gt 1) { $pass = $CommandParts[1..($CommandParts.Count - 1)] }
  & $exe @pass
  exit $LASTEXITCODE
} finally {
  if ($null -ne $fileStream) {
    $fileStream.Close()
    $fileStream.Dispose()
  }
  Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
