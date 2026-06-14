@echo off
echo ============================================================
echo  Psychological Dialogue Analysis Dashboard
echo ============================================================
echo.

REM Activate the venv
call "%~dp0psych-graph-env\Scripts\activate.bat"

REM Launch the Streamlit app
echo Starting dashboard... (browser opens automatically)
echo Press Ctrl+C to stop.
echo.
streamlit run "%~dp0app.py" --server.port 8501 --theme.base dark
