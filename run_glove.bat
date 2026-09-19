@echo off
rem Start AirPen with the glove. Double-click this file to run it.
rem Close Thonny first: only one program can use the glove's COM port.
cd /d "%~dp0"
set INPUT_BACKEND=glove
set GLOVE_SERIAL_PORT=COM4
set OCR_BACKEND=local
rem Record every glove sample to glove_logs\ for debugging.
set GLOVE_LOG=1
airwrite_env\Scripts\python.exe airwrite.py
pause
