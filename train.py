import argparse
import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


# ─────────────────────────────── CONFIG ──────────────────────────────────────

DATASET_DIR  = Path("dataset")
DATA_YAML    = DATASET_DIR / "data.yaml"
MODEL_BASE   = "yolov8n.pt"       # nano — fastest; swap for yolov8s/m/l for more accuracy
PROJECT_DIR  = Path("/content/runs/detect")
RUN_NAME     = "animal_detector"

DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ  = 640
DEFAULT_BATCH  = 4                # lower if GPU OOM; set -1 for auto-batch


# ─────────────────────────────── VALIDATION ──────────────────────────────────

def validate_dataset() -> None:
    """Sanity-check that generate_assets.py has been run first."""
    if not DATA_YAML.exists():
        print(f"[ERROR] {DATA_YAML} not found.")
        print("        Run  python generate_assets.py  first.")
        sys.exit(1)

    img_dir = DATASET_DIR / "images" / "train"
    lbl_dir = DATASET_DIR / "labels" / "train"

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    labels = list(lbl_dir.glob("*.txt"))

    if not images:
        print(f"[ERROR] No images found in {img_dir}")
        sys.exit(1)

    print(f"[OK] Dataset found: {len(images)} image(s), {len(labels)} label(s)")


# ─────────────────────────────── TRAINING ────────────────────────────────────

def train(epochs: int, imgsz: int, batch: int) -> YOLO:
    """
    Fine-tune YOLOv8n on the generated dataset.
    Returns the trained model.
    """
    print(f"\n[TRAIN] Starting YOLOv8n fine-tune for {epochs} epoch(s) …")
    print(f"        data   = {DATA_YAML}")
    print(f"        imgsz  = {imgsz}")
    print(f"        batch  = {batch}")

    model = YOLO(MODEL_BASE)

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,           # overwrite previous run with same name
        patience=10,             # early stop if no improvement for 10 epochs
        augment=True,            # YOLOv8 built-in mosaic / HSV augmentation
        cos_lr=True,             # cosine LR schedule
        optimizer="AdamW",
        lr0=1e-3,
        weight_decay=5e-4,
        save_period=5,           # checkpoint every 5 epochs
        plots=True,              # save PR/F1/confusion matrix plots
        verbose=True,
    )

    print("\n[TRAIN] Training complete.")
    return model


# ─────────────────────────────── CHECKPOINT HELPER ──────────────────────────

def resolve_checkpoint() -> Path:
    """
    Return the best available checkpoint after training.
    YOLOv8 only writes best.pt when validation runs and improves.
    With tiny datasets (train==val) it sometimes skips it, so we
    fall back to last.pt which is always written.
    """
    weights_dir = PROJECT_DIR / RUN_NAME / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"

    # Show everything that was actually saved
    found = sorted(weights_dir.glob("*.pt")) if weights_dir.exists() else []
    print(f"[INFO] Weights folder: {weights_dir}")
    if found:
        print(f"[INFO] Checkpoints found: {[f.name for f in found]}")
    else:
        print("[WARN] Weights folder is empty or missing.")

    if best.exists():
        print(f"[INFO] Using best checkpoint : {best}")
        return best
    elif last.exists():
        print(f"[INFO] best.pt not saved — using last.pt instead")
        print("       (Normal when val==train on a tiny dataset)")
        return last
    elif found:
        # Use whatever .pt exists — e.g. epoch5.pt from save_period
        fallback = found[-1]
        print(f"[INFO] Using fallback checkpoint: {fallback}")
        return fallback
    else:
        print(f"[ERROR] No .pt checkpoint found in {weights_dir}")
        print("        Training likely crashed before saving any weights.")
        print("        Check the training output above for errors.")
        sys.exit(1)


# ─────────────────────────────── EVALUATION ──────────────────────────────────

def evaluate(model: YOLO) -> None:
    """Run validation and print key metrics."""
    print("\n[EVAL] Running validation …")
    metrics = model.val(data=str(DATA_YAML), verbose=False)

    box = metrics.box
    print("\n── Detection Metrics ─────────────────────────────────────")
    print(f"  mAP@0.50       : {box.map50:.4f}")
    print(f"  mAP@0.50:0.95  : {box.map:.4f}")
    print(f"  Precision      : {box.mp:.4f}")
    print(f"  Recall         : {box.mr:.4f}")
    print("──────────────────────────────────────────────────────────")


# ─────────────────────────────── INFERENCE ───────────────────────────────────

def run_inference(model: YOLO) -> None:
    """
    Run the trained model on every training image and print detections.
    Saves annotated images to runs/detect/animal_detector/predictions/.
    """
    img_dir   = DATASET_DIR / "images" / "train"
    pred_dir  = PROJECT_DIR / RUN_NAME / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"\n[INFER] Running inference on {len(images)} image(s) …\n")

    for img_path in images:
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            iou=0.45,
            save=False,         # we'll save manually below
            verbose=False,
        )

        r = results[0]
        annotated = r.plot()    # numpy BGR array with drawn boxes

        out_path = pred_dir / img_path.name
        Image.fromarray(annotated[..., ::-1]).save(out_path)   # BGR→RGB

        # Print summary for this image
        boxes = r.boxes
        if boxes is not None and len(boxes):
            for box in boxes:
                cls_id  = int(box.cls[0])
                conf    = float(box.conf[0])
                xyxy    = box.xyxy[0].tolist()
                cls_name = r.names[cls_id]
                print(f"  {img_path.name}  →  class='{cls_name}' conf={conf:.2f}  "
                      f"bbox=[{xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f}]")
        else:
            print(f"  {img_path.name}  →  no detections (conf≥0.25)")

    print(f"\n  Annotated images saved to: {pred_dir}")


# ─────────────────────────────── EXPORT ──────────────────────────────────────

def export_model(ckpt: Path) -> None:
    """Export a checkpoint to ONNX (cross-platform deployment)."""
    print(f"\n[EXPORT] Exporting {ckpt} to ONNX …")
    export_model_obj = YOLO(str(ckpt))
    export_path = export_model_obj.export(format="onnx", imgsz=DEFAULT_IMGSZ)
    print(f"[EXPORT] Saved: {export_path}")


# ─────────────────────────────── CLI ─────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on generated animal dataset")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help=f"Input image size (default: {DEFAULT_IMGSZ})")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"Batch size; -1=auto (default: {DEFAULT_BATCH})")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip post-training inference visualisation")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip ONNX export")
    return parser.parse_args()


# ─────────────────────────────── MAIN ────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  YOLOv8 Animal Detector — Training Pipeline")
    print("=" * 60)

    # 1. Validate
    validate_dataset()

    # 2. Train
    model = train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)

    # Load best available checkpoint for evaluation/inference
    ckpt = resolve_checkpoint()
    model = YOLO(str(ckpt))

    # 3. Evaluate
    evaluate(model)

    # 4. Inference on training images
    if not args.skip_inference:
        run_inference(model)

    # 5. Export
    if not args.skip_export:
        export_model(ckpt)

    print("\n✅ All done!")
    print(f"   Checkpoints : {PROJECT_DIR / RUN_NAME / 'weights'}")
    print(f"   Predictions : {PROJECT_DIR / RUN_NAME / 'predictions'}")
    print(f"   Plots       : {PROJECT_DIR / RUN_NAME}")


if __name__ == "__main__":
    main()