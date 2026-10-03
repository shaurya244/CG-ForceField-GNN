@echo off
REM Compiles the LaTeX report using the local Tectonic engine
cd /d "%~dp0"
echo ========================================================
echo Compiling EE798R LaTeX Research Report...
echo ========================================================
..\bin\tectonic.exe report.tex
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] report.pdf successfully compiled!
) else (
    echo.
    echo [ERROR] Compilation failed.
)
pause
