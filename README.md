# Synthetic Animal Dataset Generator + YOLO Training Pipeline

A complete toolkit for generating synthetic animal datasets using Stable Diffusion and training YOLOv8 detection models.

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

## License

