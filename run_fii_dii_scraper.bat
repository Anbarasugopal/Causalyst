@echo off
REM Script to run the FII/DII daily scraper
cd /d "%~dp0"
python causalyst\src\fii_dii_scraper_TEMPLATE.py
echo.
echo NOTE: You can add this script to Windows Task Scheduler to run daily after market close.
pause
