@echo off
call D:\master\setenv.bat

SET APP_DIR=%cd%\..

SET BUILD_TAG=%APP_DIR%\.venv

echo script_info: creating virtual environment %BUILD_TAG%

python -m venv %BUILD_TAG%

%BUILD_TAG%/Scripts/activate &&^
python -m pip install pip-tools &&^
python -m pip install -U pip-tools &&^
echo script_info:success, creating virtual environment &&^
python -m piptools compile --upgrade --no-emit-index-url requirements.all.in -o requirements.txt &&^
echo script_info:setting up requirements &&^
python -m pip install -r requirements.txt &&^
echo script_info:setup complete see previous log for any errors





