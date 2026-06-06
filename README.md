# Factory Synthetic Dataset Generator + YOLO Training

## Project Structure

Create the following folder structure:

```text
project/
│
├── synth.py
├── requirements.txt
│
└── factory_assets/
    ├── bg_factory_1.jpg
    ├── bg_factory_2.jpg
    ├── bg_factory_3.jpg
    │
    ├── fg_worker.png
    └── fg_forklift.png
```

---

## Asset Naming Requirements

### Background Images

All background images must begin with:

```text
bg_
```

Examples:

```text
bg_factory_1.jpg
bg_factory_2.png
bg_warehouse.jpg
```

### Foreground Images

All foreground images must begin with:

```text
fg_
```

Examples:

```text
fg_worker.png
fg_forklift.png
```

Foreground assets should have transparent backgrounds (RGBA PNG).

---

## Class Mapping

Update the class mapping inside `synth.py` if you add new foreground assets.

Example:

```python
ASSET_CLASSES = {
    "fg_worker.png": 0,
    "fg_forklift.png": 1,
}
```

and

```python
CLASS_NAMES = [
    "worker",
    "forklift"
]
```

The filenames and class names must stay synchronized.

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running

```bash
python synth.py \
    --assets ./factory_assets \
    --data ./factory_dataset
```

Windows:

```bash
python synth.py --assets .\factory_assets --data .\factory_dataset
```

---

## Generated Output

After execution:

```text
factory_dataset/
│
├── images/
│   ├── train/
│   └── val/
│
└── labels/
    ├── train/
    └── val/
```

YOLO configuration file:

```text
dataset.yaml
```

Training runs:

```text
runs/
└── factory_run/
```

---

## Colab Notes

Google Colab storage is temporary.

After training, download:

```text
factory_dataset/
runs/
dataset.yaml
```

or save them to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

and write outputs into:

```text
/content/drive/MyDrive/
```

Otherwise all generated images, labels, and trained weights will be deleted when the Colab runtime resets.

