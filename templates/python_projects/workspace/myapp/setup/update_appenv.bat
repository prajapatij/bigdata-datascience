@echo off

call D:\master\setenv.bat

SET APP_DIR=%cd%\..

SET BUILD_TAG=%APP_DIR%\.venv

%BUILD_TAG%/Scripts/activate &&^
python -m piptools compile --upgrade --no-emit-index-url requirements.all.in -o requirements.txt &&^
echo script_info:setting up requirements &&^
python -m pip install -r requirements.txt &&^
echo script_info:setup complete see previous log for any errors