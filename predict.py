import argparse
import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


# ─────────────────────────────── CONFIG ──────────────────────────────────────

DEFAULT_WEIGHTS = Path("/content/runs/detect/animal_detector/weights/best.pt")
DEFAULT_CONF    = 0.25
DEFAULT_IOU     = 0.45
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ─────────────────────────────── INFERENCE ───────────────────────────────────

def predict_single(image_path: Path, model: YOLO, conf: float, iou: float, save: bool) -> dict:
    """
    Run inference on one image.
    Returns a result dict for summary printing.
    """
    results = model.predict(source=str(image_path), conf=conf, iou=iou, verbose=False)
    r     = results[0]
    boxes = r.boxes

    if boxes is None or len(boxes) == 0:
        print(f"  {image_path.name:<40}  →  no detection (conf≥{conf:.0%})")
        return {"file": image_path.name, "class": None, "conf": None}

    best_idx   = int(boxes.conf.argmax())
    best_cls   = int(boxes.cls[best_idx])
    best_conf  = float(boxes.conf[best_idx])
    class_name = r.names[best_cls]

    print(f"  {image_path.name:<40}  →  This is a {class_name} image  "
          f"(conf: {best_conf:.1%})")

    if save:
        out_path = image_path.parent / f"{image_path.stem}_predicted{image_path.suffix}"
        Image.fromarray(r.plot()[..., ::-1]).save(out_path)
        print(f"  {'':40}     saved → {out_path.name}")

    return {"file": image_path.name, "class": class_name, "conf": best_conf}


def predict_folder(folder: Path, model: YOLO, conf: float, iou: float, save: bool) -> None:
    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])

    if not images:
        print(f"[ERROR] No images found in {folder}")
        print(f"        Supported formats: {', '.join(IMAGE_EXTENSIONS)}")
        sys.exit(1)

    print(f"\n[INFO] Found {len(images)} image(s) in {folder}\n")
    print(f"  {'File':<40}     Prediction")
    print(f"  {'-'*40}     {'-'*30}")

    results = [predict_single(img, model, conf, iou, save) for img in images]

    # ── Summary ───────────────────────────────────────────────────────────────
    detected    = [r for r in results if r["class"] is not None]
    no_detect   = [r for r in results if r["class"] is None]
    class_counts = {}
    for r in detected:
        class_counts[r["class"]] = class_counts.get(r["class"], 0) + 1

    print(f"\n── Summary {'─'*50}")
    print(f"  Total images   : {len(results)}")
    print(f"  Detected       : {len(detected)}")
    print(f"  No detection   : {len(no_detect)}")
    if class_counts:
        print(f"  Breakdown      :")
        for cls, count in sorted(class_counts.items()):
            print(f"    {cls:<20} {count} image(s)")
    print(f"{'─'*62}")


# ─────────────────────────────── CLI ─────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and classify animals using the trained YOLOv8 model"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Path to a single input image")
    group.add_argument("--folder", help="Path to a folder of images")

    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help=f"Path to trained weights (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"Confidence threshold 0-1 (default: {DEFAULT_CONF})")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU,
                        help=f"IoU threshold for NMS (default: {DEFAULT_IOU})")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated images alongside originals")
    return parser.parse_args()


# ─────────────────────────────── MAIN ────────────────────────────────────────

if __name__ == "__main__":
    args    = parse_args()
    weights = Path(args.weights)

    if not weights.exists():
        print(f"[ERROR] Model weights not found: {weights}")
        print("        Run  python train.py  first.")
        sys.exit(1)

    print(f"[INFO] Loading model from: {weights}")
    model = YOLO(str(weights))

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"[ERROR] Image not found: {image_path}")
            sys.exit(1)
        print()
        r = predict_single(image_path, model, args.conf, args.iou, args.save)
        if r["class"]:
            print(f"\nThis is a {r['class']} image  (conf: {r['conf']:.1%})")

    elif args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"[ERROR] Folder not found: {folder}")
            sys.exit(1)
        predict_folder(folder, model, args.conf, args.iou, args.save)