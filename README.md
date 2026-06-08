# Factory Synthetic Dataset Generator + YOLO Training

A two-script toolkit for generating synthetic factory datasets and training YOLO models.

- **v1 (`synth.py`)** — composites real foreground assets onto local background images, then trains YOLOv11
- **v2 (`synth_sd.py`)** — generates images from scratch using Stable Diffusion 1.5 (standalone image gen, no YOLO training step)

---

## What's New in v2

| | v1 (`synth.py`) | v2 (`synth_sd.py`) |
|---|---|---|
| **Background source** | Local `bg_*.jpg/png` asset files | Stable Diffusion 1.5 (text-to-image) |
| **Class coverage** | Fixed to foreground assets in `factory_assets/` | Unlimited — fully prompt-driven |
| **Output** | Full YOLO dataset + trains YOLOv11 | Single generated image saved to file |
| **CLI flags** | `--assets`, `--data` | `--word`, `--output` |
| **YOLO training** | Built-in (yolo11m.pt, 15 epochs) | Not included |
| **GPU requirement** | Optional (CUDA used if available) | Required — loads model in `float16` to CUDA |
| **VRAM** | Minimal | ~4–6 GB (attention slicing enabled for 8 GB cards) |
| **New dependencies** | — | `diffusers`, `transformers` |

---

## Project Structure

```
project/
│
├── synth.py              # v1 — composite + YOLO train (ImageNet-style)
├── synth_sd.py           # v2 — Stable Diffusion image generation
├── requirements.txt
│
└── factory_assets/       # used by v1 only
    ├── bg_factory_1.jpg  # background images (bg_ prefix)
    ├── bg_factory_2.jpg
    ├── fg_worker.png     # foreground objects, transparent PNG (fg_ prefix)
    └── fg_forklift.png
```

> `synth_sd.py` does not use `factory_assets/` — all content is generated from the `--word` prompt.

---

## Asset Naming (v1 only)

### Backgrounds
Must start with `bg_`, `.jpg` or `.png`:
```
bg_factory_1.jpg
bg_warehouse.png
```

### Foregrounds
Must start with `fg_`, transparent RGBA `.png`:
```
fg_worker.png
fg_forklift.png
```

### Class Mapping
Update `ASSET_CLASSES` and `CLASS_NAMES` in `synth.py` when adding new foreground objects:

```python
ASSET_CLASSES = {
    "fg_worker.png": 0,
    "fg_forklift.png": 1,
}

CLASS_NAMES = ["worker", "forklift"]
```

---

## Installation

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

```bash
pip install -r requirements.txt
```

> For CUDA-enabled PyTorch (recommended for v2), install it before the rest:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> pip install -r requirements.txt
> ```

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
```

**v2-specific:**
- `diffusers` — runs the `StableDiffusionPipeline` (CompVis/stable-diffusion-v1-4)
- `transformers` — required by Diffusers for the text encoder/tokenizer

> `openai-clip` listed in the repo is not used by either script currently and can be omitted.

---

## Running

### v2 — Stable Diffusion (current)

Generates a single image from a text prompt and saves it to a file.

```bash
python synth_sd.py --word "factory floor with concrete walls" --output scene.png
```

**Windows:**
```bash
python synth_sd.py --word "factory floor with concrete walls" --output scene.png
```

On first run, model weights (~4 GB) are downloaded automatically from Hugging Face to `~/.cache/huggingface/`. Requires a CUDA-capable GPU.

---

### v1 — ImageNet-style composite + YOLO train (legacy)

Generates a full YOLO dataset (100 train / 20 val images) by compositing foreground assets onto local backgrounds, then immediately kicks off YOLOv11 training.

```bash
python synth.py \
    --assets ./factory_assets \
    --data ./factory_dataset
```

**Windows:**
```bash
python synth.py --assets .\factory_assets --data .\factory_dataset
```

---

## Generated Output (v1)

```
factory_dataset/
├── images/
│   ├── train/      # 100 composite images
│   └── val/        # 20 composite images
└── labels/
    ├── train/      # YOLO-format .txt label files
    └── val/

dataset.yaml        # written one level above --data

runs/
└── factory_run/    # YOLOv11 training output
```

---

## Troubleshooting

**`synth_sd.py` crashes on startup:** The script calls `pipe.to("cuda")` unconditionally. A CUDA GPU is required; there is no CPU fallback in v2.

**Out of VRAM:** `enable_attention_slicing()` is already applied, targeting 8 GB cards. If you still run out, reduce inference steps or switch to a smaller SD variant.

**`dataset.yaml` written to wrong location:** v1 writes the YAML to `../dataset.yaml` (one directory above `--data`). Make sure the parent directory exists or adjust the path in `synth.py`.

**No backgrounds found (v1):** Ensure background files in `factory_assets/` are named with the `bg_` prefix and are `.jpg` or `.png`.