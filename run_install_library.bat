@echo off
title Transformer NLP Project

echo ==========================================
echo   Transformer NLP - 20 Newsgroups
echo ==========================================
echo.

REM Move to the folder where this .bat file is located
cd /d %~dp0

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Starting training...
python train.py

echo.
echo ==========================================
echo Training finished.
pause
