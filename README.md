# Factory Synthetic Dataset Generator + YOLO Training

A tool for generating synthetic factory datasets for YOLO training. **v2** replaces the ImageNet-based background/foreground compositing approach with **Stable Diffusion 1.5** to generate photorealistic background scenes directly from text prompts, enabling open-ended class coverage beyond ImageNet's ~1,000 categories.

---

## What's New in v2

| | v1 (`synth.py`) | v2 (`synth_sd.py`) |
|---|---|---|
| **Background source** | ImageNet images (scraped/local) | Stable Diffusion 1.5 (generated) |
| **Class coverage** | ~1,000 ImageNet classes | Unlimited — prompt-driven |
| **Background variety** | Fixed to available assets | Dynamically generated per run |
| **GPU requirement** | Optional | Recommended (CUDA); CPU fallback supported |
| **Key new deps** | — | `diffusers`, `transformers`, `openai-clip` |
| **Execution script** | `synth.py` | `synth_sd.py` |
| **Prompt support** | None | `--prompt` flag for scene description |

---

## Project Structure

```
project/
│
├── synth.py              # v1 — ImageNet-based (legacy)
├── synth_sd.py           # v2 — Stable Diffusion 1.5 (current)
├── requirements.txt
│
└── factory_assets/
    ├── fg_worker.png
    └── fg_forklift.png
```

> **Note:** v2 no longer requires pre-collected background images (`bg_*.jpg`). Backgrounds are generated at runtime via SD 1.5. Foreground assets (`fg_*.png`) with transparent backgrounds are still required.

---

## Asset Naming Requirements for ImangeNet (legacy)

### Foreground Images

All foreground images must begin with `fg_` and use transparent-background RGBA PNGs:

```
fg_worker.png
fg_forklift.png
```

### Background Images (v1 only)

If using the legacy `synth.py`, background images must begin with `bg_`:

```
bg_factory_1.jpg
bg_warehouse.jpg
```

---

## Class Mapping

Update `ASSET_CLASSES` and `CLASS_NAMES` in `synth_sd.py` to match your foreground assets:

```python
ASSET_CLASSES = {
    "fg_worker.png": 0,
    "fg_forklift.png": 1,
}

CLASS_NAMES = [
    "worker",
    "forklift"
]
```

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> **GPU note:** `torch` and `torchvision` will install CPU-only by default via pip. For CUDA support, install PyTorch separately first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```
> Then run `pip install -r requirements.txt`.

---

## Requirements

```
ultralytics>=8.3.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
PyYAML>=6.0
diffusers>=2.6
transformers>=4.30.0
openai-clip>=2.0.0
```

**v2-specific packages:**

- `diffusers` — Hugging Face Diffusers library; runs the SD 1.5 pipeline
- `transformers` — required by Diffusers for tokenization and model loading
- `openai-clip` — CLIP model used for image-text alignment scoring / filtering

---

## Running

### v2 — Stable Diffusion (current)

```bash
python synth_sd.py \
    --assets ./factory_assets \
    --data ./factory_dataset \
    --prompt "industrial factory floor with concrete walls and overhead lighting"
```

**Windows:**
```bash
python synth_sd.py --assets .\factory_assets --data .\factory_dataset --prompt "industrial factory floor"
```

On first run, SD 1.5 model weights (~4 GB) will be downloaded automatically from Hugging Face to `~/.cache/huggingface/`.

### v1 — ImageNet-based (legacy)

```bash
python synth.py \
    --assets ./factory_assets \
    --data ./factory_dataset
```

---

## Generated Output

```
factory_dataset/
│
├── images/
│   ├── train/
│   └── val/
│
└── labels/
    ├── train/
    └── val/

dataset.yaml

runs/
└── factory_run/
```

---

## Troubleshooting

**Out of VRAM during generation:** SD 1.5 requires ~4–6 GB VRAM. Add `--half` flag if supported, or the script will fall back to CPU (slow but functional).

**Model download fails:** Ensure you have internet access on first run. Weights are cached after the initial download.

**`openai-clip` import error:** Try `pip install git+https://github.com/openai/CLIP.git` if the PyPI package conflicts with your environment.