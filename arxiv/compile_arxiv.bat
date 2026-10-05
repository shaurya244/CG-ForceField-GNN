@echo off
REM Compile the arXiv preprint LaTeX source using local Tectonic engine
cd /d "%~dp0"
echo ========================================================
echo Compiling arXiv Research Paper (Tectonic Engine)...
echo ========================================================

..\bin\tectonic.exe --keep-intermediates main.tex
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] main.pdf and main.bbl generated successfully!
    echo PDF location: %~dp0main.pdf
) else (
    echo.
    echo [ERROR] Compilation failed.
)
pause
