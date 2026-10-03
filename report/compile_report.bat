@echo off
REM Compiles both LaTeX reports using the local Tectonic engine
cd /d "%~dp0"
echo ========================================================
echo Compiling EE798R LaTeX Reports (Tectonic Engine)...
echo ========================================================

echo.
echo [1/2] Compiling Single-Column Linear Report (report_single_column.tex)...
..\bin\tectonic.exe report_single_column.tex
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] report_single_column.pdf compiled successfully!
) else (
    echo [WARNING] Single-column compilation failed.
)

echo.
echo [2/2] Compiling Two-Column IEEE Report (report.tex)...
if not exist build mkdir build
..\bin\tectonic.exe --outdir build report.tex
if %ERRORLEVEL% EQU 0 (
    copy /Y build\report.pdf report.pdf >nul 2>&1
    echo [SUCCESS] report.pdf compiled successfully!
) else (
    echo [WARNING] Two-column compilation failed.
)

echo.
echo ========================================================
echo Compilation complete. Check report/ directory for PDFs.
echo ========================================================
pause
