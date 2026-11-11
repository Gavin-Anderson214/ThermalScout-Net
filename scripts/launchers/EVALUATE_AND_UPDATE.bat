@echo off
cd /d "%~dp0..\.."
py -3.12 project_manager.py evaluate
pause
