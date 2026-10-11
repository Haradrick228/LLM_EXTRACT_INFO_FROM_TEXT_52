<#.
  Снимок прогона B3 каждые N секунд в лог-файл (удобно смотреть tail -f / Get-Content -Wait).

  Пример:
    .\scripts\watch_b3_status.ps1 -RunDir "eval\runs\20260512_141451_b3_full_corpus_rerank"
    .\scripts\watch_b3_status.ps1   # берёт самый новый eval\runs\*_b3_full_corpus_rerank
#>
param(
  [string] $RunDir = "",
  [int] $IntervalSec = 300,
  [string] $OutLog = "DATA\runs\b3_status_watch.log"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent

if (-not $RunDir) {
  $latest = Get-ChildItem -Path (Join-Path $root "eval\runs") -Directory -Filter "*_b3_full_corpus_rerank" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $latest) { throw "Не найден каталог eval\runs\*_b3_full_corpus_rerank — укажите -RunDir" }
  $RunDir = $latest.FullName
} else {
  $RunDir = Join-Path $root $RunDir
}

$outPath = Join-Path $root $OutLog
New-Item -ItemType File -Path $outPath -Force | Out-Null

function Write-StatusLine {
  param([string]$msg)
  $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line = "[$ts] $msg"
  Add-Content -Path $outPath -Value $line -Encoding utf8
  Write-Host $line
}

Write-StatusLine "watch_b3_status: run_dir=$RunDir interval_sec=$IntervalSec out=$OutPath"

while ($true) {
  if (-not (Test-Path $RunDir)) {
    Write-StatusLine "ERR: run_dir missing"
    Start-Sleep -Seconds $IntervalSec
    continue
  }

  $done = (Get-ChildItem -Path $RunDir -Recurse -File -Filter "retrieval_*.json" -ErrorAction SilentlyContinue).Count
  $trace = (Get-ChildItem -Path $RunDir -Recurse -File -Filter "retrieval_trace_*.jsonl" -ErrorAction SilentlyContinue).Count
  $summary = Join-Path $RunDir "summary.json"
  $sumOk = Test-Path $summary

  $python = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "run_full_corpus_rerank_matrix" }
  $pyCount = @($python).Count

  Write-StatusLine "retrieval_json_done=$done trace_jsonl=$trace summary_json=$sumOk python_b3_orchestrator_processes=$pyCount"

  if ($sumOk) {
    Write-StatusLine "DONE: summary.json exists — остановка наблюдателя."
    break
  }

  Start-Sleep -Seconds $IntervalSec
}
