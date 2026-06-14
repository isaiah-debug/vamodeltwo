@echo off
REM ============================================================
REM  VIDEO ANALYSIS STACK INSTALLER
REM  Installs WhisperX + pyannote + InsightFace + MediaPipe
REM  into psych-graph-env (RTX 5060 / CUDA 12.8)
REM ============================================================
echo.
echo ============================================================
echo  Installing video analysis stack...
echo  GPU: RTX 5060  CUDA: 12.8+
echo ============================================================
echo.

call "%~dp0psych-graph-env\Scripts\activate.bat"

echo [1/5] Installing ffmpeg-python...
pip install ffmpeg-python

echo.
echo [2/5] Installing WhisperX (GPU transcription)...
pip install whisperx

echo.
echo [3/5] Installing pyannote.audio (speaker diarization)...
pip install pyannote.audio

echo.
echo [4/5] Installing InsightFace (face detection + recognition)...
pip install insightface onnxruntime-gpu

echo.
echo [5/5] Installing MediaPipe + sklearn...
pip install mediapipe scikit-learn

echo.
echo ============================================================
echo  DONE. One more step required:
echo.
echo  1. Get a FREE HuggingFace token at: https://huggingface.co
echo  2. Accept terms at:
echo     https://huggingface.co/pyannote/speaker-diarization-3.1
echo     https://huggingface.co/pyannote/segmentation-3.0
echo  3. Set environment variable:
echo     set HF_TOKEN=your_token_here
echo     (or add to .env file)
echo.
echo  Then run:
echo     python scripts/process_videos.py --config configs/project_template.yaml
echo ============================================================
pause
