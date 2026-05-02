@echo off
REM --- Qwen3 llama.cpp server launcher
REM --- Set MODEL_PATH to your .gguf file before running

set MODEL_PATH=..\models\Qwen3.6-35B-A3B-UD-Q5_K_M.gguf
set GPU_LAYERS=999
set CONTEXT_SIZE=180224
set PORT=1234

..\llama.cpp\build\bin\Release\llama-server.exe ^
  -m %MODEL_PATH% ^
  -ngl %GPU_LAYERS% ^
  -c %CONTEXT_SIZE% ^
  -ctk q4_0 -ctv q4_0 -fa on ^
  --no-mmap ^
  --jinja ^
  -b 4096 -ub 4096 ^
  -t 16 --parallel 1 ^
  --temp 0.6 ^
  --top-p 0.95 ^
  --repeat-penalty 1.05 ^
  --port %PORT% ^
  --host 127.0.0.1
