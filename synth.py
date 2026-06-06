
import os
import gc
import yaml
import random
import torch
import argparse
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from ultralytics import YOLO
from ultralytics.utils import LOGGER

ASSET_CLASSES = {
    "fg_worker.png": 0,
    "fg_forklift.png": 1,
}

CLASS_NAMES = ["worker", "forklift"]

class RealAssetSyntheticGenerator:
    def __init__(self, asset_dir, output_dir):
        self.asset_dir = Path(asset_dir)
        self.output_dir = Path(output_dir)

        for split in ["train", "val"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        self.backgrounds = list(self.asset_dir.glob("bg_*.jpg")) + list(self.asset_dir.glob("bg_*.png"))
        self.foregrounds = list(self.asset_dir.glob("fg_*.png"))

        assert len(self.backgrounds) > 0, f"No background images (starting with 'bg_') found in {asset_dir}"
        assert len(self.foregrounds) > 0, f"No foreground images (starting with 'fg_') found in {asset_dir}"

    def build_scene(self, split="train", img_id=0):
        bg_path = random.choice(self.backgrounds)
        bg = Image.open(bg_path).convert("RGBA")
        bg_w, bg_h = bg.size

        labels_list = []
        num_objects = random.randint(1, 3)

        for _ in range(num_objects):
            fg_path = random.choice(self.foregrounds)
            cid = ASSET_CLASSES.get(fg_path.name, 0)

            fg = Image.open(fg_path).convert("RGBA")
            scale = random.uniform(0.15, 0.40)
            new_h = int(bg_h * scale)
            new_w = int(fg.width * (new_h / fg.height))
            fg = fg.resize((new_w, new_h), Image.Resampling.LANCZOS)

            paste_x = random.randint(0, bg_w - new_w)
            paste_y = random.randint(int(bg_h * 0.3), bg_h - new_h)

            mask = fg.split()[3].filter(ImageFilter.GaussianBlur(radius=1.0))
            bg.paste(fg, (paste_x, paste_y), mask=mask)

            x_center = (paste_x + (new_w / 2.0)) / bg_w
            y_center = (paste_y + (new_h / 2.0)) / bg_h
            norm_w = new_w / bg_w
            norm_h = new_h / bg_h

            labels_list.append(f"{cid} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        bg = bg.convert("RGB")
        bg = ImageEnhance.Brightness(bg).enhance(random.uniform(0.85, 1.15))
        bg = ImageEnhance.Contrast(bg).enhance(random.uniform(0.9, 1.1))

        img_name = f"factory_synth_{img_id:05d}"
        bg.save(self.output_dir / "images" / split / f"{img_name}.jpg", "JPEG")
        (self.output_dir / "labels" / split / f"{img_name}.txt").write_text("\n".join(labels_list))

    def generate_dataset(self, train_count=100, val_count=20):
        for i in range(train_count): self.build_scene(split="train", img_id=i)
        for i in range(val_count):   self.build_scene(split="val", img_id=i)

        yaml_data = {
            "path": str(self.output_dir), "train": "../images/train", "val": "../images/val",
            "nc": len(CLASS_NAMES), "names": CLASS_NAMES
        }
        with open("../dataset.yaml", "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=str, default="../datagen/factory_assets")
    parser.add_argument("--data", type=str, default="../datagen/factory_dataset")
    args = parser.parse_args()

    generator = RealAssetSyntheticGenerator(asset_dir=args.assets, output_dir=args.data)
    generator.generate_dataset(train_count=100, val_count=20)

    gc.collect()
    torch.cuda.empty_cache()

    model = YOLO("yolo11m.pt")
    model.train(
        data="../dataset.yaml", epochs=15, imgsz=640, batch=16,
        amp=True, cache=False, workers=2, nbs=64, project="../runs", name="factory_run"
    )