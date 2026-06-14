import random
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
import torch
from diffusers import StableDiffusionPipeline
from rembg import remove as rembg_remove


# ─────────────────────────────── CONFIG ──────────────────────────────────────

# Classes: {class_id: class_name}
CLASSES = {
    0: "fox",
    1: "leopard",
    2: "sheep",
}

# Prompts per class — add as many poses as you like
ANIMAL_PROMPTS = {
    0: [
        "a red fox sitting on grass, high detail, DSLR photo, white background",
        "a red fox standing alert, looking forward, DSLR photo, white background",
        "a red fox lying down, relaxed, DSLR photo, white background",
        "a red fox walking, side profile, DSLR photo, white background",
        "a red fox close-up portrait, DSLR photo, white background",
    ],
    1: [
        "a leopard sitting, high detail, DSLR photo, white background",
        "a leopard standing alert, DSLR photo, white background",
        "a leopard walking, side profile, DSLR photo, white background",
        "a leopard lying down, DSLR photo, white background",
        "a leopard close-up portrait, DSLR photo, white background",
        "a leopard crouching, DSLR photo, white background",
    ],
    2: [
        "a sheep standing, high detail, DSLR photo, white background",
        "a sheep grazing, DSLR photo, white background",
        "a sheep close-up portrait, DSLR photo, white background",
        "a sheep walking, side profile, DSLR photo, white background",
    ],
}

ANIMAL_NEG_PROMPT = "blurry, watermark, text, multiple animals, cartoon"

BACKGROUND_PROMPTS = [
    "a sunny forest clearing, photorealistic landscape, no animals",
    "a grassy meadow at golden hour, photorealistic, no animals",
    "a rocky hillside with sparse trees, photorealistic, no animals",
    "a snowy woodland path, photorealistic, no animals",
    "an autumn forest floor with fallen leaves, photorealistic, no animals",
    "a riverbank with pebbles and grass, photorealistic, no animals",
    "a countryside field at dusk, photorealistic, no animals",
    "a sunny savanna, photorealistic, no animals",
    "a dense jungle, photorealistic, no animals",
    "a dry scrubland, photorealistic, no animals",
]

BACKGROUND_NEG_PROMPT = "people, animals, text, watermark"

# Output canvas size
CANVAS_W, CANVAS_H = 640, 640

# How large the pasted foreground should be relative to the canvas (fraction)
FG_SCALE_MIN, FG_SCALE_MAX = 0.20, 0.50

# Images generated per class  (total = NUM_AUGMENTED_IMAGES * len(CLASSES))
NUM_AUGMENTED_IMAGES = 1000

SEED = 42

OUTPUT_DIR = Path("dataset")


# ─────────────────────────────── SD PIPELINE ─────────────────────────────────

def load_sd_pipeline(model_id: str = "CompVis/stable-diffusion-v1-4") -> StableDiffusionPipeline:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"[SD] Loading {model_id} on {device} ({dtype}) …")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def generate_image(
    pipe: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str = "",
    size: tuple = (512, 512),
    seed: int = SEED,
) -> Image.Image:
    # SD requires negative_prompt to match the type of prompt
    if isinstance(prompt, list) and isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt] * len(prompt)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=size[0],
        height=size[1],
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
    )
    return result.images[0]


# ─────────────────────────────── IMAGE HELPERS ───────────────────────────────

def remove_background(img: Image.Image) -> Image.Image:
    return rembg_remove(img)


def crop_to_content(rgba: Image.Image) -> Image.Image:
    bbox = rgba.getbbox()
    return rgba.crop(bbox) if bbox else rgba


def resize_foreground(fg: Image.Image, canvas_w: int, canvas_h: int) -> Image.Image:
    scale    = random.uniform(FG_SCALE_MIN, FG_SCALE_MAX)
    target_w = int(canvas_w * scale)
    target_h = int(target_w * (fg.height / fg.width))
    return fg.resize((target_w, target_h), Image.LANCZOS)


# ─────────────────────────────── AUGMENTATION ────────────────────────────────

def build_fg_augmenter() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),
        A.Rotate(limit=25, border_mode=0, p=0.6),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.4),
        A.MotionBlur(blur_limit=3, p=0.3),
    ])


def augment_foreground(fg_rgba: Image.Image, augmenter: A.Compose, idx: int) -> Image.Image:
    rgb   = np.array(fg_rgba.convert("RGB"))
    alpha = np.array(fg_rgba.split()[-1])

    random.seed(SEED + idx)
    np.random.seed(SEED + idx)
    aug_rgb = augmenter(image=rgb)["image"]

    result_rgba = Image.fromarray(aug_rgb).convert("RGBA")

    random.seed(SEED + idx)
    np.random.seed(SEED + idx)
    aug_alpha = augmenter(image=np.stack([alpha] * 3, axis=-1))["image"][:, :, 0]

    r, g, b, _ = result_rgba.split()
    return Image.merge("RGBA", (r, g, b, Image.fromarray(aug_alpha)))


# ─────────────────────────────── COMPOSITE + LABEL ───────────────────────────

def composite_and_label(
    bg: Image.Image,
    fg_rgba: Image.Image,
    canvas_w: int,
    canvas_h: int,
    idx: int,
    class_id: int,
) -> tuple:
    canvas = bg.copy().convert("RGB").resize((canvas_w, canvas_h), Image.LANCZOS)
    fg     = resize_foreground(fg_rgba, canvas_w, canvas_h)
    fg_w, fg_h = fg.size

    random.seed(SEED + idx * 13)
    x_offset = random.randint(0, max(canvas_w - fg_w, 0))
    y_offset  = random.randint(0, max(canvas_h - fg_h, 0))

    canvas.paste(fg, (x_offset, y_offset), mask=fg.split()[3])

    fg_alpha = np.array(fg.split()[3])
    rows = np.any(fg_alpha > 10, axis=1)
    cols = np.any(fg_alpha > 10, axis=0)

    if rows.any() and cols.any():
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]
        abs_x1 = x_offset + c_min
        abs_y1 = y_offset + r_min
        abs_x2 = x_offset + c_max
        abs_y2 = y_offset + r_max
    else:
        abs_x1, abs_y1 = x_offset, y_offset
        abs_x2, abs_y2 = x_offset + fg_w, y_offset + fg_h

    cx = ((abs_x1 + abs_x2) / 2) / canvas_w
    cy = ((abs_y1 + abs_y2) / 2) / canvas_h
    bw = (abs_x2 - abs_x1) / canvas_w
    bh = (abs_y2 - abs_y1) / canvas_h

    label = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    return canvas, label


# ─────────────────────────────── YOLO DATA.YAML ──────────────────────────────

def write_data_yaml(output_dir: Path, classes: dict) -> None:
    names_str = "[" + ", ".join(f"'{n}'" for n in classes.values()) + "]"
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/train\n\n"
        f"nc: {len(classes)}\n"
        f"names: {names_str}\n"
    )
    print(f"[YAML] Written: {yaml_path}")


# ─────────────────────────────── MAIN ────────────────────────────────────────

def main() -> None:
    img_dir = OUTPUT_DIR / "images" / "train"
    lbl_dir = OUTPUT_DIR / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_sd_pipeline()
    augmenter = build_fg_augmenter()

    # ── Step 1: Generate all backgrounds (shared across classes) ──────────────
    print(f"\n[1/3] Generating {len(BACKGROUND_PROMPTS)} background(s) …")
    bg_list = []
    for bi, prompt in enumerate(BACKGROUND_PROMPTS):
        print(f"      [{bi+1}/{len(BACKGROUND_PROMPTS)}] {prompt[:60]} …")
        bg = generate_image(pipe, prompt=prompt,
                            negative_prompt=BACKGROUND_NEG_PROMPT,
                            size=(768, 512), seed=SEED + 99 + bi)
        bg.save(OUTPUT_DIR / f"background_{bi}.jpg")
        bg_list.append(bg)

    # ── Step 2 & 3: Per-class foreground generation + compositing ────────────
    total_classes = len(CLASSES)
    for class_id, class_name in CLASSES.items():
        prompts = ANIMAL_PROMPTS[class_id]
        print(f"\n[2/3] Class {class_id} '{class_name}' — generating {len(prompts)} foreground(s) …")

        fg_list = []
        for fi, prompt in enumerate(prompts):
            print(f"      [{fi+1}/{len(prompts)}] {prompt[:60]} …")
            fg_raw = generate_image(
                pipe, prompt=prompt,
                negative_prompt=ANIMAL_NEG_PROMPT,
                size=(512, 512),
                seed=SEED + class_id * 100 + fi,
            )
            fg_raw.save(OUTPUT_DIR / f"foreground_raw_{class_id}_{fi}.jpg")
            fg_rgba = remove_background(fg_raw)
            fg_rgba = crop_to_content(fg_rgba)
            fg_rgba.save(OUTPUT_DIR / f"foreground_{class_id}_{fi}.png")
            fg_list.append(fg_rgba)
            print(f"             saved foreground_{class_id}_{fi}.png")

        print(f"\n[3/3] Class {class_id} '{class_name}' — compositing {NUM_AUGMENTED_IMAGES} image(s) …")
        for i in range(NUM_AUGMENTED_IMAGES):
            fg_rgba = fg_list[i % len(fg_list)]
            bg      = bg_list[i % len(bg_list)]
            aug_fg  = augment_foreground(fg_rgba, augmenter, idx=class_id * 10000 + i)

            composite, label = composite_and_label(
                bg, aug_fg, CANVAS_W, CANVAS_H,
                idx=class_id * 10000 + i,
                class_id=class_id,
            )

            img_name = f"sample_{class_id}_{i:04d}.jpg"
            lbl_name = f"sample_{class_id}_{i:04d}.txt"
            composite.save(img_dir / img_name, quality=95)
            (lbl_dir / lbl_name).write_text(label + "\n")

            if (i + 1) % 100 == 0 or i == 0:
                print(f"      [{i+1}/{NUM_AUGMENTED_IMAGES}] {img_name}  →  {label}")

    # ── Write data.yaml ───────────────────────────────────────────────────────
    write_data_yaml(OUTPUT_DIR, CLASSES)

    total = NUM_AUGMENTED_IMAGES * len(CLASSES)
    print(f"\n✅ Dataset generation complete!  ({total} total images)")
    print(f"   Classes : {list(CLASSES.values())}")
    print(f"   Images  : {img_dir}")
    print(f"   Labels  : {lbl_dir}")
    print(f"   Config  : {OUTPUT_DIR / 'data.yaml'}")
    print("\nNext step: run  python train.py")


if __name__ == "__main__":
    main()