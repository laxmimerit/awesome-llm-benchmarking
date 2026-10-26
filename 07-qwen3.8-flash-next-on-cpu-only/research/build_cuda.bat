@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%APPDATA%\Python\Python314\site-packages\ninja\data\bin;%PATH%"
cd /d "%~dp0"
echo === cl version ===
cl 2>&1 | findstr /C:"Version"
if exist build-cuda rmdir /s /q build-cuda
echo === configure ===
cmake llama.cpp -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 || exit /b 1
echo === build ===
cmake --build build-cuda --target llama-cli llama-server llama-bench -j 20 || exit /b 1
echo === BUILD_OK ===
