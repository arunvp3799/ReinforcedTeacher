@echo off
REM Quick test script for APPS Model Analysis
REM This runs the analysis with 3 examples for quick testing

echo ========================================
echo APPS Model Analysis - Quick Test
echo ========================================
echo.

echo Step 1: Testing setup...
python test_setup.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Setup test failed. Please check the errors above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Step 2: Running analysis with 3 examples...
echo ========================================
echo.

python run_analysis.py --num-examples 3 --output apps_quick_test.json
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Analysis failed. Please check the errors above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Step 3: Viewing results...
echo ========================================
echo.

python view_results.py apps_quick_test.json --summary

echo.
echo ========================================
echo Quick test complete!
echo ========================================
echo.
echo Results saved to: apps_quick_test.json
echo.
echo Next steps:
echo   - View specific example: python view_results.py apps_quick_test.json --example 1
echo   - Compare outputs: python view_results.py apps_quick_test.json --compare 1
echo   - Export to markdown: python view_results.py apps_quick_test.json --export-md report.md
echo.
pause
