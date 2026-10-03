# PowerShell compilation script for LaTeX report using local Tectonic engine
$ErrorActionPreference = "Stop"
$ReportDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ReportDir

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Compiling EE798R LaTeX Research Report..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$TectonicPath = Join-Path $ReportDir "..\bin\tectonic.exe"
if (-not (Test-Path $TectonicPath)) {
    throw "Tectonic engine not found at $TectonicPath"
}

& $TectonicPath report.tex
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] report.pdf successfully compiled at $(Join-Path $ReportDir 'report.pdf')!" -ForegroundColor Green
} else {
    Write-Host "`n[ERROR] Compilation failed." -ForegroundColor Red
}
