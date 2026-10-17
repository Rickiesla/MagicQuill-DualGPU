@echo off
:: ============================================
:: MagicQuill AMD GPU Setup (RX 580 / 8GB VRAM)
:: Uses DirectML instead of CUDA
:: Funciona com o venv existente (sem Conda)
:: ============================================

cd /d "%~dp0"

echo.
echo === MagicQuill - Setup AMD GPU (DirectML) ===
echo.

:: ---- 1. Verificar venv ----
if not exist "venv\Scripts\python.exe" (
    echo [ERRO] Nao encontrei o 'venv'. Crie primeiro com:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    pause
    exit /b 1
)

echo [1/4] Usando venv existente em: venv\Scripts\python.exe

:: ---- 2. Remover PyTorch CUDA e instalar DirectML ----
echo.
echo [2/4] Trocando PyTorch CUDA -^> DirectML...
echo   (removendo versao cu118...)
venv\Scripts\pip.exe uninstall -y torch torchvision torchaudio torchsde 2>nul

echo   (instalando PyTorch CPU + torch-directml...)
venv\Scripts\pip.exe install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
if %ERRORLEVEL% NEQ 0 (
    echo [ERRO] Falha ao instalar PyTorch CPU.
    pause
    exit /b 1
)

venv\Scripts\pip.exe install torch-directml
if %ERRORLEVEL% NEQ 0 (
    echo [ERRO] Falha ao instalar torch-directml.
    echo   Verifique se o driver AMD (Adrenalin) esta atualizado e se voce esta no Windows 10+.
    pause
    exit /b 1
)
venv\Scripts\pip.exe install torchsde==0.2.6

:: ---- 3. Testar DirectML ----
echo.
echo [3/4] Testando DirectML...
venv\Scripts\python.exe -c "import torch; import torch_directml; d = torch_directml.device(); print('  DirectML OK! Placa detectada:', torch_directml.device_name(0))"
if %ERRORLEVEL% NEQ 0 (
    echo [ERRO] DirectML nao detectou sua GPU AMD.
    echo   - Atualize o driver AMD Adrenalin
    echo   - Confirme que e Windows 10 ou 11
    pause
    exit /b 1
)

:: ---- 4. Verificar modelos ----
echo.
echo [4/4] Verificando modelos...
if not exist "models" (
    echo.
    echo [AVISO] A pasta 'models' NAO EXISTE!
    echo   Voce precisa baixar os modelos (~12 GB) antes de rodar.
    echo   Siga as instrucoes no README.md do MagicQuill (secao de download dos modelos).
    echo.
    echo   Os arquivos devem ficar em:
    echo     models\checkpoints\SD1.5\realisticVisionV60B1_v51VAE.safetensors
    echo     models\controlnet\control_v11p_sd15_scribble.safetensors
    echo     models\controlnet\color_finetune.safetensors
    echo     models\inpaint\brushnet\random_mask_brushnet_ckpt\diffusion_pytorch_model.safetensors
    echo     models\llava-v1.5-7b-finetune-clean\  (pasta completa)
    echo.
    pause
    exit /b 0
)

:: ---- Iniciar ----
echo.
echo === Tudo pronto! Iniciando MagicQuill com DirectML... ===
echo.
set USE_DIRECTML=1
set AMD_GPU=1
venv\Scripts\python.exe gradio_run.py

echo.
echo MagicQuill foi encerrado.
pause
