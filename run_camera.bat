@echo off
rem Start AirPen in camera mode. Double-click this file to run it.
rem Press V inside AirPen to switch to the glove (close Thonny first).
cd /d "%~dp0"
set INPUT_BACKEND=camera
rem The phone's camera address is kept in camera_address.txt. Press I in
rem camera mode to type a new one when you change Wi-Fi network.
rem "auto" finds the glove's USB port by itself on any computer.
set GLOVE_SERIAL_PORT=auto
set OCR_BACKEND=local
rem Record every glove sample to glove_logs\ for debugging.
set GLOVE_LOG=1
rem Record every camera frame to camera_logs\ for debugging.
set AIRPEN_LOG=1
airwrite_env\Scripts\python.exe airwrite.py
pause
