# Поднимает Chroma + Postgres из infra/docker-compose.yml (без telegram app — он в profile telegram).
# Ollama в этом compose нет: при необходимости: docker start ollama
$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$scraped = Join-Path $RepoRoot "scraped_site"
if (-not (Test-Path $scraped)) {
    New-Item -ItemType Directory -Path $scraped | Out-Null
    Set-Content -Path (Join-Path $scraped ".gitkeep") -Value ""
}

Write-Host "[up_infra] repo: $RepoRoot" -ForegroundColor Cyan
docker compose -f infra/docker-compose.yml up -d chroma-init postgres chroma
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[up_infra] Chroma heartbeat:" -ForegroundColor Cyan
curl.exe -sS --max-time 10 "http://127.0.0.1:18000/api/v2/heartbeat"
Write-Host ""
