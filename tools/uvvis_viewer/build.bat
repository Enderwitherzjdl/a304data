:: build.bat  （在 VS Code 终端里直接：build）
@echo off
cd /d %~dp0

pyinstaller ^
  --onefile ^
  --windowed ^
  --icon=harumasa.ico ^
  --name UVVisViewer ^
  --paths=..\.. ^
  main.py
