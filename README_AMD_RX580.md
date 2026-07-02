# MagicQuill — Guia para AMD RX 580 (8GB VRAM) via DirectML

Este guia explica como rodar o MagicQuill em uma **AMD RX 580** (ou outras placas AMD
Polaris/Vega/RDNA não suportadas pelo ROCm no Windows) usando **DirectML**.

> ⚠️ **Importante:** A RX 580 não é suportada pelo ROCm (apenas no Linux e apenas para
> arquiteturas mais novas). No Windows, a única saída é o **DirectML** da Microsoft.

---

## ✅ O que funciona

| Recurso | Funciona? | Observação |
|---------|-----------|------------|
| Edição de imagem (BrushNet + SD1.5) | ✅ Sim | Roda em **fp32** (mais lento que NVIDIA, mas funciona) |
| ControlNets (scribble, color) | ✅ Sim | Carregados junto com o modelo principal |
| Interface Gradio | ✅ Sim | Sem alterações |

## ⚠️ O que NÃO funciona (ou é limitado)

| Recurso | Funciona? | Observação |
|---------|-----------|------------|
| **Draw and Guess** (LLaVA) | ⚠️ Lento | Roda na **CPU** (não há bitsandbytes para AMD). Cada inferência pode levar **30s–2min**. |
| Velocidade geral | ⚠️ Lento | DirectML + fp32 é consideravelmente mais lento que CUDA+fp16 |

---

## 📦 Pré-requisitos

1. **Windows 10/11** com driver AMD atualizado (Adrenalin mais recente).
2. **Python 3.10** via Conda (Miniconda ou Anaconda).
3. **Git** instalado.
4. ~**20 GB livres** em disco (modelos + dependências).
5. Os modelos do MagicQuill já baixados (coloque-os em `models/` conforme o README original).

---

## 🚀 Instalação passo a passo

### Opção A — Script automático (recomendado)

```bat
windows_setup_amd.bat
```

O script cria o environment Conda, instala PyTorch+DirectML e inicia o MagicQuill.

### Opção B — Manual

#### 1. Criar environment Conda
```bat
conda create -n MagicQuill python=3.10 -y
conda activate MagicQuill
conda init cmd.exe
```
Reabra o terminal após o `conda init`.

#### 2. Instalar PyTorch (CPU) + DirectML
```bat
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
pip install torch-directml
```

Verifique se o DirectML detectou sua RX 580:
```bat
python -c "import torch_directml; print(torch_directml.device_name(0))"
```
Você deve ver algo como `AMD Radeon RX 580 Series`.

#### 3. Instalar dependências do MagicQuill
```bat
pip install gradio_magicquill-0.0.1-py3-none-any.whl
copy /y pyproject.toml MagicQuill\LLaVA
pip install -e MagicQuill\LLaVA\
pip install -r requirements.txt
```

#### 4. Rodar
```bat
set USE_DIRECTML=1
set AMD_GPU=1
python gradio_run.py
```

Ao iniciar, você verá no console:
```
[MagicQuill] DirectML enabled for AMD GPU support.
```

---

## 🔧 O que foi modificado no código

As seguintes alterações foram feitas para suportar DirectML/AMD:

| Arquivo | Alteração |
|---------|-----------|
| `MagicQuill/llava_new.py` | Removido `load_4bit=True` (bitsandbytes é só CUDA). LLaVA roda em **CPU** com fp32. `.cuda()` → `.to(self.model.device)`. |
| `MagicQuill/brushnet_nodes.py` | Quando DirectML está ativo, usa `load_state_dict` em vez de `load_checkpoint_and_dispatch` (accelerate não suporta DirectML). |
| `MagicQuill/scribble_color_edit.py` | Detecta DirectML e usa **fp32** em vez de fp16 (fp16 é instável em Polaris). |
| `MagicQuill/comfy/options.py` | Lê `USE_DIRECTML`/`AMD_GPU` e habilita o parsing de args. |
| `MagicQuill/comfy/model_management.py` | Força `LOW_VRAM` no DirectML; protege `torch.backends.cuda` com try/except. |
| `gradio_run.py` | Injeta `--directml` no `sys.argv` quando `USE_DIRECTML=1`. |

---

## 🐛 Solução de problemas

### "torch_directml not found" / device_name vazio
- Reinstale: `pip uninstall torch-directml -y && pip install torch-directml`
- Atualize o driver AMD (Adrenalin ≥ 23.x).
- No Windows 7/8 o DirectML **não funciona** (precisa Win10+).

### Erro de memória (OOM) / imagem preta
- A RX 580 tem 8GB. Em fp32 tudo pesa o dobro. Se ocorrer OOM:
  - Reduza a resolução da imagem (ex: 768×768 em vez de 1024×1024).
  - Feche outros apps que usem VRAM (navegador, jogos).
- **Imagem preta** geralmente indica NaN por causa de fp16 — já ajustado para fp32 automaticamente.

### LLaVA muito lento (Draw and Guess)
- O LLaVA roda na CPU e pode levar até 2 minutos por inferência. Para desativar:
  - Na interface, clique no ícone de **varinha mágica** (wand) para desligar o
    preenchimento automático de prompt, e digite o prompt manualmente.

### Erro `device_map` / `accelerate`
- Já corrigido: o BrushNet agora carrega via `load_state_dict` no DirectML.

### O DirectML não está sendo usado (uso de CPU 100%)
- Confirme as variáveis de ambiente antes de rodar:
  ```bat
  echo %USE_DIRECTML%
  ```
  Deve imprimir `1`.

---

## 📊 Expectativa de performance (RX 580 8GB, fp32)

| Operação | Tempo aproximado |
|----------|------------------|
| Carregamento inicial dos modelos | 30–60s |
| Inferência LLaVA (Draw & Guess, CPU) | 30s – 2min |
| Geração de imagem (50 steps, 512×512) | 30s – 90s |
| Geração de imagem (50 steps, 1024×1024) | 2 – 5 min |

> Para comparação, uma RTX 3060 (CUDA, fp16) faz a mesma geração em ~5s.
> DirectML + fp32 é ~10–20× mais lento que CUDA+fp16. É o preço de não ter ROCm.

---

## ❓ FAQ

**Posso usar fp16 para ir mais rápido?**
Não na RX 580 (Polaris). O fp16 via DirectML costuma gerar imagens pretas (NaN)
nessa arquitetura. GPUs RDNA2+ (RX 6000) têm suporte melhor a fp16.

**Vai melhorar no futuro?**
O ROCm para Windows (HIP SDK) ainda não suporta Polaris. Quando suportar,
será muito mais rápido que DirectML. Por enquanto, DirectML é o caminho.

**Funciona com outras placas AMD?**
Sim — qualquer GPU que o DirectML detectar (RX 400/500/5000/6000/7000).
Placas RX 6000+ em fp16 podem ser bem mais rápidas.
