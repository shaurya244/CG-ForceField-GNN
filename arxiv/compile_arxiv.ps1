# PowerShell compilation script for arXiv preprint using local Tectonic engine
$ErrorActionPreference = "Stop"
$ArxivDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ArxivDir

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Compiling arXiv Research Paper (Tectonic Engine)..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$TectonicPath = Join-Path $ArxivDir "..\bin\tectonic.exe"
if (-not (Test-Path $TectonicPath)) {
    Write-Error "Tectonic engine not found at $TectonicPath"
    exit 1
}

& $TectonicPath --keep-intermediates main.tex
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] main.pdf and main.bbl generated successfully!" -ForegroundColor Green
    Write-Host "PDF location: $(Join-Path $ArxivDir 'main.pdf')" -ForegroundColor Green
} else {
    Write-Host "`n[ERROR] Compilation failed." -ForegroundColor Red
    exit 1
}
