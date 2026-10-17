# 🪶 MagicQuill-DualGPU

**Fork de [`magic-quill/MagicQuill`](https://github.com/magic-quill/MagicQuill)** — sistema
interativo de edição de imagens com inteligência artificial *(CVPR 2025)*,
**adaptado por [@Rickiesla](https://github.com/Rickiesla) para rodar em GPU AMD
(RX 580, DirectML)** e mantido 100% compatível com NVIDIA (CUDA).

[![Fork de magic-quill/MagicQuill](https://img.shields.io/badge/fork-magic--quill%2FMagicQuill-blue.svg)](https://github.com/magic-quill/MagicQuill)
[![Autor da adaptação AMD: Rickiesla](https://img.shields.io/badge/adapta%C3%A7%C3%A3o%20AMD-Rickiesla-blue.svg)](https://github.com/Rickiesla)
[![arXiv](https://img.shields.io/badge/arXiv-2411.09703-b31b1b.svg)](https://arxiv.org/abs/2411.09703)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![GPU: AMD (DirectML) / NVIDIA (CUDA)](https://img.shields.io/badge/GPU-AMD%20DirectML%20%7C%20NVIDIA%20CUDA-green.svg)](#2-compatibilidade-de-gpu)

> ⚡ **Este repositório é um fork modificado** do projeto oficial
> [magic-quill/MagicQuill](https://github.com/magic-quill/MagicQuill) — ele **não**
> é o repositório dos autores. **A adaptação para GPU AMD (DirectML) foi feita por
> [@Rickiesla](https://github.com/Rickiesla)** (esta fork); o projeto original só
> suportava NVIDIA/CUDA. As diferenças em relação ao original:
>
> - ✅ **Suporte AMD/DirectML** — adaptado para a **AMD RX 580 (6 GB VRAM)** por
>   [@Rickiesla](https://github.com/Rickiesla), que roda normalmente nesta placa;
> - ✅ **Compatibilidade NVIDIA mantida** — mesmo código roda em CUDA (verificado
>   em **NVIDIA RTX 3060 12 GB**);
> - ✅ **Documentação nova em português** (este README + guia AMD) com download
>   dos modelos, instalação das duas marcas e solução de problemas;
> - ✅ **Correção de bugs** (steps fracionário, prompt nulo, caminhos de modelo
>   errados nos `.bat`, `print.warning` inexistente) e limpeza de código morto.
>
> **Testado em:** AMD RX 580 6 GB (DirectML) e NVIDIA RTX 3060 12 GB (CUDA).
> Créditos do projeto original, paper e licença continuam sendo dos autores
> originais (ver [seção 14](#14-licença-e-citação)).

> O MagicQuill transforma traços desenhados à mão em edições precisas de imagem.
> Basta desenhar sobre a foto, escrever um prompt e deixar a IA preencher o resto —
> adicionar elementos, remover detalhes ou recolorir regiões com controle local.

Projeto original: [magicquill.art](https://magicquill.art/) ·
Repositório oficial: [github.com/magic-quill/MagicQuill](https://github.com/magic-quill/MagicQuill) ·
Demo oficial: [Hugging Face](https://huggingface.co/spaces/AI4Editing/MagicQuill) ·
Vídeo: [YouTube](https://www.youtube.com/watch?v=5DiKfONMnE4)

![Exemplo de edição](docs/gifs/deer.gif)

---

## Índice

1. [O que é o projeto](#1-o-que-é-o-projeto)
2. [Compatibilidade de GPU](#2-compatibilidade-de-gpu)
3. [Requisitos do sistema](#3-requisitos-do-sistema)
4. [Modelos de IA — download e instalação](#4-modelos-de-ia--download-e-instalação)
5. [Instalação — NVIDIA (CUDA)](#5-instalação--nvidia-cuda)
6. [Instalação — AMD (DirectML)](#6-instalação--amd-directml)
7. [Como rodar](#7-como-rodar)
8. [Interface e parâmetros](#8-interface-e-parâmetros)
9. [Draw and Guess (LLaVA)](#9-draw-and-guess-llava)
10. [Estrutura técnica do código](#10-estrutura-técnica-do-código)
11. [Desempenho esperado](#11-desempenho-esperado)
12. [Solução de problemas](#12-solução-de-problemas)
13. [Pontos de atenção (análise estática do código)](#13-pontos-de-atenção-análise-estática-do-código)
14. [Licença e citação](#14-licença-e-citação)

---

## 1. O que é o projeto

O MagicQuill é um **editor de imagens assistido por IA**, publicado no CVPR 2025
por pesquisadores da HKUST, Ant Group, ZJU e HKU. Ele combina três modelos de
deep learning (um modelo base SD1.5, dois ControlNets e um BrushNet) com uma
interface web desenhada para desenho à mão:

| Pincel | O que faz |
|--------|-----------|
| **Adicionar** (traço verde) | Insere novos elementos/detalhes guiados pelo prompt, seguindo o seu traço |
| **Remover** (traço vermelho) | Apaga ou redesenha a área marcada |
| **Colorir** (pincel de cor) | Pinta uma região e o ControlNet de cor espalha a paleta pela imagem |

Além disso, o botão **Draw and Guess** (varinha mágica) usa um modelo LLaVA-7B
para *escrever o prompt automaticamente* a partir dos seus traços — **desligado
por padrão** por ser pesado (ver [seção 9](#9-draw-and-guess-llava)).

**Fluxo de trabalho:** suba uma imagem → desenhe os traços → (opcional) gere o
prompt → ajuste os parâmetros → **Run** → a imagem editada aparece, com opção de
salvar automaticamente.

> Neste repositório o projeto foi adaptado para rodar em **duas GPU**:
> NVIDIA via **CUDA** e AMD via **DirectML** (API da Microsoft — é o caminho
> possível para placas AMD no Windows, já que o ROCm não suporta Polaris).

---

## 2. Compatibilidade de GPU

| GPU | Modo de execução | Comando | Situação |
|-----|------------------|---------|----------|
| **NVIDIA** (qualquer com 8 GB+) | CUDA (fp16/fp8) | `venv\Scripts\python.exe gradio_run.py` | ✅ Verificado abrindo e carregando os modelos em uma **RTX 3060 12 GB** |
| **AMD Radeon RX 580 (6 GB VRAM)** | DirectML (fp32) | `rodar.bat` | ✅ **Rodava normalmente** nesta GPU (era a placa usada antes da RTX 3060) |
| Outras AMD (RX 400/500/5000/6000/7000) | DirectML (fp32) | `rodar.bat` | ✅ Desde que o DirectML detecte a placa (Win10/11 + driver atual) |
| Sem GPU | CPU | — | ⛔ Possível, porém imprestável para geração |

Notas importantes:

- **CUDA e DirectML são caminhos de arranque diferentes** — o mesmo comando não
  serve para as duas marcas (detalhes na [seção 7](#7-como-rodar)).
- A RX 580 **não é suportada pelo ROCm no Windows**; o DirectML é a única via.
  Em DirectML o modelo roda em **fp32** (o fp16 gera imagens pretas/NaN em
  GPUs Polaris), por isso é naturalmente mais lento que CUDA.
- O LLaVA (Draw and Guess) roda **na CPU**, em qualquer placa
  (`MagicQuill/llava_new.py` fixa `LLAVA_DEVICE = "cpu"`).

### Como o programa escolhe a GPU: CUDA × DirectML

O CUDA **não fica no código** — o vínculo está em três camadas:

| Camada | Onde vive | Papel |
|--------|-----------|-------|
| **Driver + GPU** | na **máquina** | Driver NVIDIA fornece o CUDA; driver AMD Adrenalin fornece DirectX 12 (base do DirectML) |
| **PyTorch** | no **`venv/`** | `torch+cu121` = build CUDA (usa NVIDIA). `torch+cpu` **+** `torch-directml` = usa qualquer GPU via DirectML. É aqui que a GPU fica "vinculada" ao ambiente |
| **Código** | nos `.py` | Genérico: na inicialização lê `USE_DIRECTML` (`gradio_run.py`) e escolhe `torch.cuda` **ou** `torch_directml`. **Nada do código precisa ser editado** para alternar |

```
rodar.bat (USE_DIRECTML=1)  →  DirectML (AMD)
sem a variável (padrão)     →  CUDA (NVIDIA)
```

- **NVIDIA:** CUDA é chamado por padrão — é só rodar `venv\Scripts\python.exe gradio_run.py`.
- **AMD:** com `rodar.bat` o CUDA **nunca é chamado** (nem é tentado); o programa
  vai direto para o DirectML e força fp32.
- ⚠️ **`rodar.bat` numa NVIDIA** → cai em DirectML (funciona, mas ~10× mais
  lento). ⚠️ **Comando "cru" numa AMD** → o CUDA não existe na placa, o
  programa cai para a **CPU**. Cada marca usa sempre o seu comando.
- **O `venv` é específico da máquina** (o `pyvenv.cfg` aponta para o Python
  local). Ao mudar de PC, **recrie o `venv`** com os passos da
  [seção 5](#5-instalação--nvidia-cuda) ou [seção 6](#6-instalação--amd-directml) —
  código e `.bat` vão juntos sem nenhuma alteração.

---

## 3. Requisitos do sistema

| Item | Mínimo | Observação |
|------|--------|------------|
| Sistema | Windows 10/11 (64 bits) | Linux também funciona (`linux_setup.sh`), mas as instruções abaixo são para Windows |
| Python | **3.10** | O `venv` deste projeto foi criado com 3.10.11; versões 3.11+ não são compatíveis com os pacotes já instalados |
| Git | Qualquer | Necessário para clonar com o submódulo LLaVA |
| GPU | 8 GB VRAM (NVIDIA) ou 6 GB (AMD/DirectML) | Ver [seção 2](#2-compatibilidade-de-gpu) |
| Driver da GPU | **Atualizado** | NVIDIA: driver GeForce · AMD: Software Adrenalin. O DirectML exige driver recente — instale/atualize **antes** de clonar |
| RAM | 16 GB | ~28 GB se for habilitar o LLaVA |
| Disco | ~30 GB livres | ~6 GB (`venv`) + ~20 GB (modelos) + projeto |

---

## 4. Modelos de IA — download e instalação

Os modelos **não vão para o GitHub** (20 GB). Após clonar o repositório, é
obrigatório baixá-los e colocá-los em `models/` exatamente nestes caminhos:

```
models/
├── checkpoints/
│   └── SD1.5/
│       └── realisticVisionV60B1_v51VAE.safetensors   # 2,0 GB  OBRIGATÓRIO
├── controlnet/
│   ├── control_v11p_sd15_scribble.safetensors        # 1,4 GB  OBRIGATÓRIO
│   └── color_finetune.safetensors                    # 1,4 GB  OBRIGATÓRIO
├── inpaint/
│   └── brushnet/
│       └── random_mask_brushnet_ckpt/
│           ├── diffusion_pytorch_model.safetensors   # 2,3 GB  OBRIGATÓRIO
│           └── config.json
├── preprocessor/
│   ├── sk_model.pth                                  # 16 MB   OBRIGATÓRIO
│   ├── sk_model2.pth                                 # 16 MB   OBRIGATÓRIO
│   └── table5_pidinet.pth                            # 2,7 MB  OBRIGATÓRIO
├── configs/                                          # 11 YAMLs do ComfyUI
└── llava-v1.5-7b-finetune-clean/                     # 13,2 GB OPCIONAL*
```

`*` O LLaVA só é usado pelo **Draw and Guess** (que já vem desligado). Sem ele,
o MagicQuill funciona por completo — **você digita o prompt na mão**.

**Total:** ≈ **7 GB** sem LLaVA · ≈ **20 GB** com LLaVA.

### Onde baixar

1. **Link oficial (SharePoint dos autores):**

   ```
   https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliucz_connect_ust_hk/EWlGF0WfawJIrJ1Hn85_-3gB0MtwImAnYeWXuleVQcukMg?e=Gcjugg&download=1
   ```

   Baixe `models.zip` pelo navegador (cerca de 20 GB) e extraia de modo que a
   pasta `models/` fique na raiz do repositório.

2. **Espelho no Hugging Face:** [huggingface.co/LiuZichen/MagicQuill-models](https://huggingface.co/LiuZichen/MagicQuill-models)
   (útil se o link da SharePoint não abrir).

> ⚠️ **Cuidado com o caminho do BrushNet:** os pesos do BrushNet ficam em
> `models/inpaint/brushnet/random_mask_brushnet_ckpt/` — **não** em
> `models/brushnet/`. Um README antigo trazia o caminho errado; seguir-o
> provoca um `KeyError` logo na inicialização. O caminho correto é o da árvore
> acima (é o que o código lê em `MagicQuill/brushnet_nodes.py`).

> 💡 Usuários da China continental podem usar o espelho:
> `set HF_ENDPOINT=https://hf-mirror.com` antes de rodar.

---

## 5. Instalação — NVIDIA (CUDA)

Testado com **Python 3.10.11 + PyTorch 2.2.1 (CUDA 12.1)** em uma RTX 3060 12 GB.

```bat
:: 1. Clonar ESTA fork com o submódulo LLaVA (o --recursive é obrigatório)
git clone --recursive https://github.com/Rickiesla/MagicQuill-DualGPU.git
cd MagicQuill-DualGPU

:: 2. Criar o ambiente virtual (venv)
python -m venv venv
venv\Scripts\activate

:: 3. Instalar o PyTorch com suporte a GPU NVIDIA
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
:: Alternativa (instrução original dos autores, CUDA 11.8):
:: pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

:: 4. Instalar a interface gráfica (wheel já vem no repositório)
pip install gradio_magicquill-0.0.1-py3-none-any.whl

:: 5. Instalar o LLaVA (necessário mesmo se você não for usar o Draw and Guess)
copy /Y pyproject.toml MagicQuill\LLaVA\
pip install -e MagicQuill\LLaVA\

:: 6. Instalar as dependências restantes
pip install -r requirements.txt

:: 7. Baixar os modelos de IA — obrigatório, eles NÃO vêm no clone (ver seção 4)
```

**Verificar se a GPU foi reconhecida:**

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Deve imprimir algo como `2.2.1+cu121 True NVIDIA GeForce RTX 3060`.

---

## 6. Instalação — AMD (DirectML)

Testado com a **AMD RX 580 (6 GB VRAM)** — *rodava normalmente nesta placa*.
A única diferença em relação à NVIDIA são os passos 3 e a forma de rodar.

```bat
:: Clonar ESTA fork (contém o suporte AMD/DirectML — o repo oficial não tem)
git clone --recursive https://github.com/Rickiesla/MagicQuill-DualGPU.git
cd MagicQuill-DualGPU
python -m venv venv
venv\Scripts\activate

:: 3a. PyTorch de CPU + plugin DirectML (é assim que funciona a AMD no Windows)
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-directml
pip install torchsde==0.2.6

:: 4, 5 e 6 — iguais à NVIDIA (wheel, LLaVA, requirements)
pip install gradio_magicquill-0.0.1-py3-none-any.whl
copy /Y pyproject.toml MagicQuill\LLaVA\
pip install -e MagicQuill\LLaVA\
pip install -r requirements.txt

:: 7. Baixar os modelos de IA — obrigatório, eles NÃO vêm no clone (ver seção 4)
```

**Verificar se a AMD foi reconhecida:**

```bat
python -c "import torch_directml; print(torch_directml.device_name(0))"
```

Deve imprimir algo como `AMD Radeon RX 580 Series`.

### Sobre o `windows_setup_amd.bat`

Ele **não cria o ambiente sozinho**: apenas troca o PyTorch CUDA por
PyTorch-CPU + `torch-directml` dentro de um `venv` **já existente** e inicia o
programa. Ou seja, rode os passos 1, 2, 4, 5 e 6 acima e use o `.bat` só para
o passo 3 — ou faça tudo manualmente, como acima.

### Sobre o `windows_setup.bat` (NVIDIA)

Script legado dos autores: cria ambiente **Conda** e instala `cu118`. Só
funciona se você tiver o Conda instalado; para o setup com `venv` descrito
neste README, siga a [seção 5](#5-instalação--nvidia-cuda) manualmente.

---

## 7. Como rodar

O programa é um servidor local (FastAPI + Gradio) que abre no navegador.

> ⚠️ **Antes de rodar:** os pesos da IA precisam estar em `models/` — se ainda
> não baixou, faça a [seção 4](#4-modelos-de-ia--download-e-instalação). Sem eles
> a inicialização falha com `KeyError`/`FileNotFoundError`.

### NVIDIA (CUDA)

```bat
venv\Scripts\activate
set USE_DIRECTML=0
venv\Scripts\python.exe gradio_run.py
```

Aparecerá no console:

```
[MagicQuill] Running in default CUDA mode. Set USE_DIRECTML=1 to enable AMD GPU via DirectML.
Device: cuda:0 NVIDIA GeForce RTX 3060 : native
INFO:     Uvicorn running on http://127.0.0.1:7860
```

### AMD (DirectML)

```bat
rodar.bat
```

(O `rodar.bat` apenas define `USE_DIRECTML=1` e chama o mesmo `gradio_run.py`.)

Aparecerá:

```
[MagicQuill] DirectML enabled for AMD GPU support.
[MagicQuill] DirectML detected: using float32 for BrushNet (fp16 unstable on AMD Polaris)
```

### Abrindo a interface

Depois que aparecer `Uvicorn running`, abra no navegador:

```
http://127.0.0.1:7860
```

O primeiro carregamento dos modelos leva de **30 s a 2 min**. Para encerrar,
feche o terminal ou pressione `Ctrl+C`.

> ⚠️ **Não use `rodar.bat` numa placa NVIDIA** — ele força o modo DirectML
> (mais lento e exige o pacote `torch-directml` instalado). Em NVIDIA, use
> sempre o comando do modo CUDA. Inversamente, para AMD use sempre
> `rodar.bat`, nunca o comando CUDA.
>
> A porta está fixada em `127.0.0.1:7860` (`gradio_run.py`) e todos os flags de
> linha de comando (`--port`, `--cpu`, `--force-fp32`…) são ignorados pelo
> código — não há como mudá-los sem editar o programa.

---

## 8. Interface e parâmetros

**Barra lateral de parâmetros (accordion *parameters*):**

| Parâmetro | Padrão | O que faz / observação |
|-----------|--------|------------------------|
| Base Model Name | `SD1.5/realisticVisionV60B1_v51VAE.safetensors` | Modelo base (só existe um por padrão) |
| Auto Save | desligado | Grava cada resultado em `output\magicquill_AAAA-MM-DD_HH-MM-SS.png` |
| Resolution | slider 512 (**uso real 384**) | Define o redimensionamento da imagem **enviada** — ajuste **antes** de subir a imagem. O valor efetivo só muda depois que você mexe no slider |
| Negative Prompt | vazio | O que a imagem **não** deve conter |
| Fine Edge | `disable` | Usa LineArt fino em vez do pré-processador de scribble |
| Grow Size | 15 | Dilata a área afetada pela máscara |
| Edge Strength | 0.55 | Força do ControlNet de traços (0–5) |
| Color Strength | 0.55 | Força do ControlNet de cor (0–5) |
| Inpaint Strength | 1.0 | Força do BrushNet (0–5) |
| Seed | −1 (aleatório) | Fixa o resultado para reproduzir a geração |
| Steps | 20 | **Use sempre números inteiros** (ver [seção 13](#13-pontos-de-atenção-análise-estática-do-código)) |
| CFG | 5.0 | Peso do prompt |
| Sampler / Scheduler | `euler_ancestral` / `karras` | Algoritmo de amostragem |

**Comportamentos que valem saber:**

- `Auto Save` e `Resolution` são **globais** — valem para todas as abas abertas
  ao mesmo tempo.
- A imagem de máscara precisa ter **canal alfa (transparência)**; sem ele a
  máscara é considerada vazia e a geração não altera nada.
- Imagem original, máscara e imagem colorida precisarão ter o **mesmo tamanho**.
- Se a geração falhar, o erro aparece só no console — a interface apenas não
  retorna imagem.

---

## 9. Draw and Guess (LLaVA)

O modelo **LLaVA-7B** escreve o prompt automaticamente a partir dos seus traços.

- **Vem DESLIGADO por padrão** (`ENABLE_LLAVA=0`), porque carrega ~13 GB de
  pesos e precisa de **~28 GB de RAM** para rodar em fp32.
- Roda **na CPU**, mesmo com NVIDIA — cada clique na varinha leva **30 s a 2 min**.

Para ligar:

```bat
set ENABLE_LLAVA=1
venv\Scripts\python.exe gradio_run.py
```

Sem ele, é só digitar o prompt no campo de texto — o restante do programa não
é afetado. O download da pasta `llava-v1.5-7b-finetune-clean/` (13,2 GB) só é
necessário se você for usar este recurso.

---

## 10. Estrutura técnica do código

```
MagicQuill/                     # núcleo do programa (usa o ComfyUI embutido)
├── comfy/                      # fork do ComfyUI (samplers, model_management, cli_args…)
│   ├── model_management.py     # seleção de device: CUDA / DirectML / CPU, VRAM, fp16/fp32/fp8
│   ├── options.py              # lê USE_DIRECTML / AMD_GPU e habilita o parsing de argumentos
│   └── samplers.py, sd.py…     # amostragem e VAE
├── brushnet/                   # implementação do BrushNet (inpainting)
├── brushnet_nodes.py           # nós: carregar/aplicar BrushNet e ControlNets
├── comfyui_utils.py            # nós de alto nível: Checkpoint, ControlNet, KSampler, VAE,
│                               #   pré-processadores (PIDINet/LineArt/Color)
├── scribble_color_edit.py      # orquestra o pipeline completo (carrega tudo no __init__)
├── llava_new.py                # wrapper do LLaVA (fixo em CPU/fp32)
├── model_patch.py              # monkeypatch do ComfyUI p/ injetar o BrushNet
├── folder_paths.py             # localização das pastas de modelos
└── LLaVA/                      # submódulo (github.com/haotian-liu/LLaVA)

gradio_run.py                   # entrada do programa: UI + rotas + servidor
gradio_magicquill-*.whl         # componente Gradio da interface (empacotado)
temp_whl/                       # código-fonte da wheel (ignorado pelo Git)
rodar.bat                       # roda em DirectML (AMD)
windows_setup*.bat, linux_setup.sh
```

**Fluxo de execução:**

1. `gradio_run.py` importa `ScribbleColorEditModel` → carrega checkpoint,
   ControlNets e BrushNet para a GPU (o console mostra "BrushNet SD1.5 model is loaded").
2. A UI Gradio monta os controles e o FastAPI expõe duas rotas locais:
   - `POST /magic_quill/guess_prompt` — chama o LLaVA (ou devolve `""` se desligado);
   - `POST /magic_quill/process_background_img` — redimensiona a imagem enviada.
3. Ao clicar em **Run**, `generate()` prepara imagens/máscaras e chama
   `ScribbleColorEditModel.process()`: pré-processadores → ControlNets →
   BrushNet → KSampler → VAE → blending → imagem final em base64.
4. `uvicorn` serve tudo em `127.0.0.1:7860`.

**Versões verificadas neste ambiente:** Python 3.10.11 · PyTorch 2.2.1 ·
Gradio 5.4.0 · diffusers 0.31.0 · transformers 4.37.2 · accelerate 0.33.0 ·
opencv-python 4.10.0.84 · `gradio_magicquill` 0.0.1.

---

## 11. Desempenho esperado

| Operação | AMD RX 580 6 GB (DirectML, fp32) | NVIDIA CUDA (fp16/fp8) |
|----------|----------------------------------|-------------------------|
| Carga inicial dos modelos | 30–90 s | 30–60 s |
| Geração 512×512, 20 steps | ~30–90 s | ~5 s |
| Geração 1024×1024 | 2–5 min | ~15–30 s |
| Draw and Guess (LLaVA, CPU) | 30 s–2 min | 30 s–2 min |

> DirectML + fp32 é algo entre 10× e 20× mais lento que CUDA + fp16 — é o
> custo de não ter ROCm no Windows. Para acelerar na AMD, reduza a resolução e
> o número de steps.

---

## 12. Solução de problemas

| Sintoma | Causa provável | Solução |
|---------|----------------|---------|
| `ImportError: No module named 'torch_directml'` ao rodar `rodar.bat` | Ambiente NVIDIA sem o pacote DirectML | Use o comando CUDA da [seção 7](#7-como-rodar) (não o `rodar.bat`) |
| `[MagicQuill] Running in default CUDA mode` mas você tem AMD | `USE_DIRECTML` não definido | Rode via `rodar.bat` |
| A imagem gerada não muda nada | Máscara sem canal alfa, ou prompt vazio | Exporte a máscara com transparência; escreva um prompt |
| `KeyError`/`TypeError` logo na inicialização | Pasta de modelos incompleta ou BrushNet no caminho errado | Conferir a árvore da [seção 4](#4-modelos-de-ia--download-e-instalação) |
| Imagem saiu toda preta | fp16 em GPU AMD | Em DirectML o código já força fp32; se continuar, feche outros programas que usem VRAM |
| `out of memory` | Resolução/steps altos demais | Reduza Resolution (ex.: 512) e feche jogos/navegador |
| Erro `0xc0000006` ou "Windows não pode acessar o arquivo …safetensors" | Falha de leitura do **disco** (setor defeituoso/cabo) | Verifique o disco (`chkdsk D: /scan`), SMART e cabo SATA — **não é erro do MagicQuill** |
| Erro `TypeError` mencionando `range()`/`linspace` ao gerar | Steps fracionário (ex.: 25.5) | Já corrigido no código (slider inteiro) — atualize se seu clone for antigo |
| Draw and Guess devolve prompt vazio | LLaVA desligado (padrão) ou RAM insuficiente | `set ENABLE_LLAVA=1` com ~28 GB de RAM |
| Downloads do Hugging Face falham | Restrição de rede | `set HF_ENDPOINT=https://hf-mirror.com` |

Guias adicionais de troubleshooting AMD (DirectML, OOM, fp16, drivers):
**[`README_AMD_RX580.md`](README_AMD_RX580.md)** — *atenção: esse guia é mais
antigo que este README e ainda menciona Conda e caminhos desatualizados; em caso
de divergência, siga este README.*

---

## 13. Pontos de atenção (análise estática do código)

Análise feita sobre o código desta fork, dividida em dois blocos: **correções
que fizemos aqui** (bugs que já existiam no projeto oficial e foram resolvidos
neste fork) e **comportamentos herdados do original** (mantidos de propósito —
reproduzem exatamente o projeto que os autores publicaram).

### 13.1 Correções aplicadas nesta fork (bug do original → resolvido aqui)

| Problema que existe no projeto original | O que fizemos aqui |
|---|---|
| Steps fracionário (ex.: `25.5`) quebrava a geração com `TypeError` dentro do sampler | Slider ganhou `step=1` e `generate()` converte `steps` para inteiro |
| Prompt inicialmente `null` chegava até o encoder | `null` vindo da interface é convertido em `""` antes do encoder |
| `except:` em `MagicQuill/model_patch.py` chamava `print.warning` (inexistente em Python) — a exceção engolia mais um erro | Captura a exceção e imprime com `print`, informando o formato dos tensores |
| Máscara total vazia (sem alfa/traços) seguia em silêncio pra geração — imagem saía igual | O console agora **avisa antes** de gerar; *pendência*: tamanhos divergentes entre imagem e máscara ainda falham, com o erro só no terminal |
| Código morto: imports sem uso (`subprocess`, `requests`, `io` duplicado), rádio `stroke_as_edge` comentado, argumento `progress` caindo na posição errada de `base_model_version` em `process()` | Removidos (nodes nunca chamados — `PowerPaint`, `CutForInpaint` — e ramos `stroke_as_edge=disable` ficaram, por segurança) |
| `temp_whl/`, `run_out.log`, `run_err.log` podiam ser versionados por engano | Adicionados ao `.gitignore` |

### 13.2 Comportamentos herdados do projeto original (mantidos — iguais ao oficial)

- **Erros engolidos pela UI** — por design, a interface não derruba por
  exceção; **sempre olhe o terminal** quando algo falhar.
- **UNet em fp8 por padrão em CUDA** (`--fp8_e4m3fn-unet` com `default=True`
  em `MagicQuill/comfy/cli_args.py`) — funciona e economiza VRAM na RTX;
  é uma diferença em relação ao ComfyUI padrão.
- **Fallback de OOM do VAE só existe para CUDA** — não mexemos (alterar o
  caminho de DirectML/AMD era arriscado); em AMD, reduza a resolução para
  evitar OOM.
- **Duas instâncias de `folder_paths`** convivem no processo — funciona, mas
  a ordem dos imports é frágil (`scribble_color_edit.py` compensa com
  `sys.path.append`).
- **Porta/host fixos** (`127.0.0.1:7860`) e flags de CLI ignoradas —
  comportamento documentado na [seção 7](#7-como-rodar).

---

## 14. Licença e citação

### Licença

O código está sob **CC BY-NC 4.0** ([LICENSE](LICENSE)) — **uso não comercial**.
(O badge antigo dizia CC BY-SA 4.0; o arquivo `LICENSE` manda.)

Ao publicar resultados derivados, inclua a atribuição aos autores originais.

> **Sobre esta fork:** a adaptação para GPU AMD (DirectML / RX 580) foi feita por
> [@Rickiesla](https://github.com/Rickiesla). O crédito do projeto original
> (arquitetura, modelo, paper) é inteiramente dos autores listados abaixo.

### Citação

```bibtex
@inproceedings{liu2025magicquill,
  title     = {MagicQuill: An Intelligent Interactive Image Editing System},
  author    = {Zichen Liu and Yue Yu and Hao Ouyang and Qiuyu Wang and Ka Leong Cheng
               and Wen Wang and Zhiheng Liu and Qifeng Chen and Yujun Shen},
  booktitle = {CVPR},
  year      = {2025},
  eprint    = {2411.09703},
  archivePrefix = {arXiv}
}
```

### Agradecimentos

Implementação baseada em
[ComfyUI-BrushNet](https://github.com/nullquant/ComfyUI-BrushNet) ·
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) ·
[LLaVA](https://github.com/haotian-liu/LLaVA) ·
[comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) ·
[fabric.js](https://github.com/fabricjs/fabric.js)

Paper original: [arXiv:2411.09703](https://arxiv.org/abs/2411.09703) ·
Site: [magicquill.art](https://magicquill.art/) ·
V2: [magicquill.art/v2](https://magicquill.art/v2/)
