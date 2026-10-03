# PowerShell compilation script for LaTeX reports using local Tectonic engine
$ReportDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ReportDir

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Compiling EE798R LaTeX Reports (Tectonic Engine)..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$TectonicPath = Join-Path $ReportDir "..\bin\tectonic.exe"
if (-not (Test-Path $TectonicPath)) {
    Write-Error "Tectonic engine not found at $TectonicPath"
    exit 1
}

Write-Host "`n[1/2] Compiling Single-Column Linear Report (report_single_column.tex)..." -ForegroundColor Yellow
& $TectonicPath report_single_column.tex
if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] report_single_column.pdf compiled successfully!" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Single-column report compilation failed." -ForegroundColor Red
}

Write-Host "`n[2/2] Compiling Two-Column IEEE Report (report.tex)..." -ForegroundColor Yellow
$BuildDir = Join-Path $ReportDir "build"
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}
& $TectonicPath --outdir build report.tex
if ($LASTEXITCODE -eq 0) {
    try {
        Copy-Item -Force "$BuildDir\report.pdf" "$ReportDir\report.pdf" -ErrorAction SilentlyContinue
    } catch {}
    Write-Host "[SUCCESS] report.pdf compiled successfully!" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Two-column report compilation failed." -ForegroundColor Red
}

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "Compilation complete. Check report/ directory for PDFs." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
