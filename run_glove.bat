@echo off
rem Start AirPen with the glove. Double-click this file to run it.
rem Close Thonny first: only one program can use the glove's COM port.
cd /d "%~dp0"
set INPUT_BACKEND=glove
rem "auto" finds the glove's USB port by itself on any computer.
set GLOVE_SERIAL_PORT=auto
set OCR_BACKEND=local
rem Record every glove sample to glove_logs\ for debugging.
set GLOVE_LOG=1
rem Record every camera frame to camera_logs\ for debugging.
set AIRPEN_LOG=1
airwrite_env\Scripts\python.exe airwrite.py
pause
