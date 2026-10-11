# Full-corpus final research pipeline (B3 + validity + report).
# Requires: Docker Desktop (Chroma on 127.0.0.1:18000), optional Ollama for answer-eval.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "== verify stack =="
python scripts/verify_local_stack_running.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARN: stack not fully up; B3/neg-control need Chroma. Continuing with offline steps."
}

Write-Host "== B3 matrix 6x3 (replay) =="
$env:RETRIEVAL_EVAL_SHUFFLE = "0"
$env:EVAL_SCENARIO_TYPE = "full_corpus"
$env:RETRIEVAL_METRIC_MODES = "substring,token_f1,semantic"
python -m eval.replay_experiment run phase_b3_full_corpus_rerank_matrix
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Latest = Get-ChildItem "eval\runs\*_b3_full_corpus_rerank" | Sort-Object Name | Select-Object -Last 1
Write-Host "Latest B3 run:" $Latest.FullName

python -m eval.diagnostics.validate_research_run_contract --run-dir $Latest.FullName
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== validity suite =="
python -m eval.diagnostics.run_final_research_validity_suite

Write-Host "== answer-eval top cells =="
python -m eval.diagnostics.run_final_research_answer_top --b3-run-dir $Latest.FullName --max-rows 0

Write-Host "== reports =="
python -m eval.reporting.emit_phase_b_final_research_report
python -m eval.reporting.build_retrieval_recovery_report

Write-Host "Done."
