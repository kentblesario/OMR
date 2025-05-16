@echo off
REM Install Python
echo Installing Python...
curl https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe --output python-installer.exe
start /wait python-installer.exe /quiet InstallAllUsers=1 PrependPath=1
del python-installer.exe

REM Update pip
echo Updating pip...
python -m pip install --upgrade pip

REM Install required libraries
echo Installing required libraries...
pip install opencv-python numpy datetime

REM Run the test.py file
echo Running test.py...
python test1.py

pause