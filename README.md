# Factory Synthetic Dataset Generator + YOLO Training

A toolkit for generating synthetic factory datasets and training YOLO models.

* **v1 (`synth.py`)** — Composites real foreground assets onto local background images, then trains YOLOv11.
* **v2 (`synth_sd.py`)** — Generates images from scratch using Stable Diffusion 1.5 (standalone image gen, no YOLO training step).
* **v3 (`modified_sd.py`)** — Advanced Stable Diffusion 1.4 generator featuring automatic compositional prompt analysis and smart parameter injection.

**v4 Features:**
- **`generate_assets.py`** — Generates synthetic animal images (fox, leopard, sheep) using StableDiffusion-v1-4, removes backgrounds with rembg, and composites them onto AI-generated backgrounds with YOLO-formatted bounding box labels
- **`train.py`** — Fine-tunes YOLOv8n on the generated dataset with built-in evaluation, inference visualization, and ONNX export
- **`predict.py`** — Run inference on single images or entire folders with confidence/IoU threshold control

---

## What's New in v4

The v4 pipeline is a **complete end-to-end solution** for synthetic animal detection dataset generation:

### Key Features

| Feature | Description |
|---------|-------------|
| **Stable Diffusion Generation** | Creates animal foregrounds + natural backgrounds from text prompts |
| **Background Removal** | Automatic RGBA masking using `rembg` for clean pasting |
| **Augmentation Pipeline** | HSV changes, rotation, noise, motion blur via `albumentations` |
| **YOLO Label Generation** | Automatic bounding box calculation in center-mode format |
| **Complete Training Loop** | Train → Evaluate → Inference → ONNX export in one script |
| **Batch Prediction** | Process single images or entire folders with summary statistics |

### Dataset Specs

- **Classes:** fox (0), leopard (1), sheep (2)
- **Canvas Size:** 640×640
- **Images per Class:** 1,000 augmented samples
- **Total Dataset:** 3,000 images
- **Label Format:** `class_id cx cy bw bh` (YOLO center-normalized)

---

## Project Structure

```text
project/
│
├── generate_assets.py    # v4 — SD-based synthetic dataset generation
├── train.py              # v4 — YOLOv8 training + evaluation + export
├── predict.py            # v4 — Inference on images/folders
├── requirements.txt
├── dataset/              # Generated after running generate_assets.py
│   ├── images/train/     # 3,000 composite images
│   ├── labels/train/     # YOLO .txt labels
│   └── data.yaml         # YOLO dataset config
│
└── runs/detect/animal_detector/  # Training outputs
    ├── weights/
        ├── best.pt
        ├── last.pt
    ├── predictions/       # Annotated inference images
    └── plots/             # PR/F1/confusion matrix plots
```

---

## Installation

1. **Set up a virtual environment:**
```bash
python -m venv .venv

# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
```

2. **Install PyTorch with CUDA support** (strongly recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. **Install remaining requirements:**
```bash
pip install -r requirements.txt

```

4. Use [Animal Generation Dataset](https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset?select=test) as follows: 
```bash
    i. If using locally : 
            import kagglehub
            path = kagglehub.dataset_download("antoreepjana/animals-detection-images-dataset")
            print("Path to dataset files:", path)
    ii. If using colab : 
            !kaggle datasets download -d antoreepjana/animals-detection-images-dataset
        followed by: 
            !unzip /content/animals-detection-images-dataset.zip -d /content/animals
```

### Full Dependencies (`requirements.txt`)

```text
ultralytics>=8.3.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
PyYAML>=6.0
diffusers>=0.21.0
transformers>=4.40.0
numpy>=1.24.0
albumentations>=1.3.0
rembg>=2.0.0
```

---

## Running the Pipeline

### Step 1: Generate Synthetic Dataset

```bash
python generate_assets.py
```

**What it does:**
- Generates 10 background images (forest, meadow, hillside, etc.)
- Creates 5 fox, 6 leopard, 4 sheep foregrounds per class
- Applies 1,000 augmentations per class with random scaling, rotation, HSV, noise
- Outputs YOLO-formatted labels with bounding boxes



---

### Step 2: Train YOLOv8 Model

```bash
python train.py
```

**Customizable options:**
```bash
python train.py --epochs 50 --imgsz 640 --batch 4
```

**What it does:**
- Fine-tunes `yolov8n.pt` on the generated dataset
- Runs validation and prints mAP@0.50, Precision, Recall
- Runs inference on all training images, saves annotated outputs
- Exports model to ONNX for cross-platform deployment


---

### Step 3: Run Predictions

**Single image:**
```bash
python predict.py --image test.jpg --save
```

**Folder of images:**
```bash
python predict.py --folder ./test_images --conf 0.25 --iou 0.45 --save
```

**Custom thresholds:**
```bash
python predict.py --image test.jpg --weights path/to/best.pt --conf 0.5 --iou 0.4
```


---

## Configuration

Edit these constants in `generate_assets.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLASSES` | `{0: "fox", 1: "leopard", 2: "sheep"}` | Class ID → name mapping |
| `ANIMAL_PROMPTS` | See script | Text prompts per class for SD generation |
| `NUM_AUGMENTED_IMAGES` | 1000 | Images per class |
| `CANVAS_W`, `CANVAS_H` | 640, 640 | Output canvas size |
| `FG_SCALE_MIN/MAX` | 0.20, 0.50 | Foreground scale relative to canvas |
| `SEED` | 42 | Random seed for reproducibility |
| `OUTPUT_DIR` | `dataset` | Output directory |



---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA not available** | Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| **Out of VRAM (~8GB required)** | `generate_assets.py` has `enable_attention_slicing()` by default. Reduce steps or image size if needed |
| **`data.yaml` not found** | Run `python generate_assets.py` first before `train.py` |
| **No `best.pt` checkpoint** | Script falls back to `last.pt` when validation doesn't improve (common with tiny datasets where train==val) |
| **rembg fails** | Ensure `rembg>=2.0.0` is installed; may require U2Net model download on first run |
| **albumentations augmentation artifacts** | Alpha channel is augmented separately to prevent color bleeding |

---


---

## What's New in v3 (`modified_sd.py`)

The newest addition, `modified_sd.py`, automatically parses your prompt for semantic composition cues (e.g., "close up" vs "wide shot" vs explicit distances like "from 100m away") and alters model generation on the fly.

### Feature Breakdown

| Feature / Metric | v1 (`synth.py`) | v2 (`synth_sd.py`) | v3 (`modified_sd.py`) **[NEW]** |
| --- | --- | --- | --- |
| **Background Source** | Local `bg_*.jpg/png` asset files | Stable Diffusion 1.5 | Stable Diffusion 1.4 |
| **Class Coverage** | Fixed to assets in `factory_assets/` | Unlimited — prompt-driven | Unlimited — prompt-driven |
| **Composition Control** | Manual anchoring | Standard generation | **Auto-detected** (`small`, `closeup`, `normal`) |
| **Smart Injection** | N/A | No | **Yes** (Injects custom framing tokens per mode) |
| **Default Dimensions** | Matching local asset sizes | $512 \times 512$ | Mode adaptive (e.g., $768 \times 512$ for `small` mode) |
| **Output** | Full YOLO dataset + trains YOLOv11 | Single image | One or more generated images + auto-naming |
| **VRAM Requirement** | Minimal | ~4–6 GB (SD 1.5) | ~4–6 GB (SD 1.4 + attention slicing) |

---

## Project Structure

```text
project/
│
├── synth.py              # v1 — Composite + YOLO train (ImageNet-style)
├── synth_sd.py           # v2 — Stable Diffusion 1.5 image generation
├── modified_sd.py        # v3 — Smart Composition Stable Diffusion 1.4 pipeline
├── requirements.txt
│
└── factory_assets/       # Used by v1 only
    ├── bg_factory_1.jpg  
    ├── bg_factory_2.jpg
    ├── fg_worker.png     # Foreground objects, transparent PNG
    └── fg_forklift.png

```

> **Note:** Neither `synth_sd.py` (v2) nor `modified_sd.py` (v3) use the `factory_assets/` directory. All content is dynamically generated via text prompts.

---

## Smart Composition Modes (v3 Only)

`modified_sd.py` checks your text input against specific keyword patterns and regular expressions to bucket the intent into one of three styles, automatically adjusting settings:

* **`small` Mode:** Triggered by words like *tiny, distant, wide shot, bird's eye*, or distance/percentage phrases (e.g., *"from 50m away"*, *"5% of the frame"*).
* *Behavior:* Automatically switches resolution to **$768 \times 512$**, boosts CFG Guidance Scale to **12.0**, injects environmental prompt modifiers, and enforces a strict macro/close-up negative prompt.


* **`closeup` Mode:** Triggered by words like *close up, macro, portrait, headshot*.
* *Behavior:* Generates at **$512 \times 512$**, uses a **7.5** CFG scale, and allows your exact prompt to dictate the shot without token injection.


* **`normal` Mode:** Default fallback when no strong compositional keywords are present.

---

## Installation

1. Set up a virtual environment:
```bash
python -m venv .venv

```


* **Windows:** `.venv\Scripts\activate`
* **Linux / macOS:** `source .venv/bin/activate`


2. **Install PyTorch with CUDA support** (strongly recommended for v2 and required for v3):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

```


3. Install the remaining requirements:
```bash
pip install -r requirements.txt

```



### Updated Dependencies (`requirements.txt`)

```text
ultralytics>=8.3.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
PyYAML>=6.0
diffusers>=2.6
transformers>=4.30.0

```

---

## Running the Toolkit

### v3 — Smart Stable Diffusion (`modified_sd.py`)

Generates images based on programmatic prompt analysis. If no output name is given, a filename is auto-generated using a safe slug of your text prompt.

* **Example (Triggers `small` subject mode automatically):**
```bash
python modified_sd.py --word "a forklift seen from 100m away on a factory floor"

```


* **Example (Generating multiple candidate variations with a specific seed):**
```bash
python modified_sd.py --word "close up photo of electronic components" --num-images 3 --seed 42

```


* **Manual Override:** You can bypass the auto-detection engine entirely using the `--mode` flag:
```bash
python modified_sd.py --word "industrial robotic arm" --mode small --output custom_robotic.png

```



### v2 — Standard Stable Diffusion 1.5 (`synth_sd.py`)

Generates a single square image using standard prompt settings.

```bash
python synth_sd.py --word "factory floor with concrete walls" --output scene.png

```

### v1 — Legacy Dataset Engine & YOLO Training (`synth.py`)

Composites local PNG files onto local backgrounds and automatically kicks off a 15-epoch training loop using `yolo11m.pt`.

```bash
python synth.py --assets ./factory_assets --data ./factory_dataset

```

---

## Troubleshooting

* **`modified_sd.py` or `synth_sd.py` crashes on startup:** Both scripts call `.to("cuda")` unconditionally. A dedicated NVIDIA GPU with CUDA is required. There is no CPU fallback.
* **Out of VRAM:** `modified_sd.py` initializes with `enable_attention_slicing()` active by default to support 8 GB VRAM cards. If you still encounter crashes, reduce the image size explicitly via the `--width` and `--height` CLI flags, or reduce your `--steps` count (default: 50).
* **Target Asset Classes Not Recognized (v1):** Remember to update the `ASSET_CLASSES` dictionary and `CLASS_NAMES` list inside `synth.py` whenever you add new foreground items to your `factory_assets/` folder.