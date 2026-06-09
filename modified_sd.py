import argparse
import re
import torch
from diffusers import StableDiffusionPipeline
 
# ---------------------------------------------------------------------------
# Keyword lists used to detect prompt intent
# ---------------------------------------------------------------------------
 
SMALL_SUBJECT_KEYWORDS = [
    "tiny", "small", "minuscule", "distant", "far away", "aerial",
    "wide shot", "wide angle", "bird's eye", "birds eye", "overhead",
    "landscape", "vast", "sprawling", "lone figure", "silhouette",
    "from above", "from a distance", "dot on", "speck",
]
 
CLOSE_UP_KEYWORDS = [
    "close up", "close-up", "macro", "portrait", "face", "headshot",
    "detailed", "zoomed in", "extreme close", "tight shot",
]
 
# Framing injected only when small-subject mode is detected
SMALL_PREFIX = (
    "wide angle shot, extreme wide shot, tiny subject, "
    "subject occupies less than 5 percent of frame, vast environment, "
)
SMALL_SUFFIX = (
    ", very far away, distant, dwarfed by surroundings, "
    "establishing shot, expansive background"
)
SMALL_NEGATIVE = (
    "close-up, macro, portrait, large subject, subject filling frame, "
    "headshot, cropped, tight shot, zoomed in"
)
 
GENERIC_NEGATIVE = (
    "blurry, low quality, distorted, watermark, text, ugly, bad anatomy"
)
 
# ---------------------------------------------------------------------------
# Prompt analysis
# ---------------------------------------------------------------------------
 
def detect_mode(prompt: str):
    """
    Returns 'small', 'closeup', or 'normal' based on prompt keywords.
    Also returns a human-readable reason string.
    """
    lower = prompt.lower()
 
    small_hits  = [kw for kw in SMALL_SUBJECT_KEYWORDS if kw in lower]
    close_hits  = [kw for kw in CLOSE_UP_KEYWORDS       if kw in lower]
 
    if small_hits and not close_hits:
        return "small",  f"small-subject keywords detected: {small_hits}"
    if close_hits and not small_hits:
        return "closeup", f"close-up keywords detected: {close_hits}"
 
    # Check for explicit scale cues: "from 100m away", "1/100 scale", etc.
    if re.search(r"from \d+\s*(m|km|meters?|feet|miles?) away", lower):
        return "small", "distance phrase detected"
    if re.search(r"\d+\s*%\s*(of the frame|of frame|of image)", lower):
        return "small", "explicit size-percentage phrase detected"
 
    return "normal", "no strong compositional cues, using generic settings"
 
 
def build_prompt_and_negative(user_prompt: str, mode: str):
    if mode == "small":
        return (
            f"{SMALL_PREFIX}{user_prompt}{SMALL_SUFFIX}",
            SMALL_NEGATIVE,
        )
    # close-up or normal — don't inject anything, let the prompt speak
    return user_prompt, GENERIC_NEGATIVE
 
 
def slugify(text: str, max_len: int = 40) -> str:
    """Turn prompt text into a safe filename fragment."""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    return text[:max_len]
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(
        description="Stable Diffusion 1.4 generator — auto-detects composition mode from prompt."
    )
    parser.add_argument("--word",   type=str, required=True,
                        help="Text prompt")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename. Auto-generated from prompt if omitted.")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="CFG scale. Auto-chosen per mode if omitted.")
    parser.add_argument("--steps",  type=int, default=50,
                        help="Denoising steps. (default: 50)")
    parser.add_argument("--width",  type=int, default=None,
                        help="Image width. Auto-chosen per mode if omitted.")
    parser.add_argument("--height", type=int, default=None,
                        help="Image height. Auto-chosen per mode if omitted.")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--num-images", type=int, default=1,
                        help="Candidates to generate. (default: 1)")
    parser.add_argument("--negative-prompt", type=str, default=None,
                        help="Override the auto-chosen negative prompt.")
    parser.add_argument("--mode",   type=str, default=None,
                        choices=["small", "closeup", "normal"],
                        help="Force a composition mode instead of auto-detecting.")
    args = parser.parse_args()
 
    print("=" * 60)
    print("Stable Diffusion 1.4 — Auto-Mode Generator")
    print("=" * 60)
 
    # --- Detect / override mode ---
    if args.mode:
        mode   = args.mode
        reason = "forced via --mode flag"
    else:
        mode, reason = detect_mode(args.word)
 
    print(f"\nMode   : {mode.upper()}")
    print(f"Reason : {reason}")
 
    # --- Per-mode defaults ---
    MODE_DEFAULTS = {
        "small":   {"guidance_scale": 12.0, "width": 768, "height": 512},
        "closeup": {"guidance_scale":  7.5, "width": 512, "height": 512},
        "normal":  {"guidance_scale":  7.5, "width": 512, "height": 512},
    }
    defaults = MODE_DEFAULTS[mode]
    guidance_scale = args.guidance_scale or defaults["guidance_scale"]
    width          = args.width          or defaults["width"]
    height         = args.height         or defaults["height"]
 
    # --- Build prompts ---
    prompt, auto_negative = build_prompt_and_negative(args.word, mode)
    negative_prompt = args.negative_prompt or auto_negative
 
    print(f"\nPrompt          : {prompt}")
    print(f"Negative prompt : {negative_prompt}")
    print(f"CFG scale       : {guidance_scale}  |  Steps: {args.steps}  "
          f"|  Size: {width}x{height}")
 
    # --- Auto output filename ---
    if args.output is None:
        slug        = slugify(args.word)
        args.output = f"{slug}_{mode}.png"
        print(f"Output          : {args.output}  (auto-generated)")
    else:
        print(f"Output          : {args.output}")
 
    # --- Load model ---
    print("\n[1/2] Loading Stable Diffusion 1.4...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    print("      Model loaded.")
 
    # --- Generate ---
    print(f"\n[2/2] Generating {args.num_images} image(s)…")
 
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
 
    results = pipe(
        prompt                = prompt,
        negative_prompt       = negative_prompt,
        guidance_scale        = guidance_scale,
        num_inference_steps   = args.steps,
        width                 = width,
        height                = height,
        num_images_per_prompt = args.num_images,
        generator             = generator,
    )
 
    # --- Save ---
    base, ext = (args.output.rsplit(".", 1) if "." in args.output
                 else (args.output, "png"))
 
    saved = []
    for i, img in enumerate(results.images):
        path = f"{base}_candidate{i+1}.{ext}" if args.num_images > 1 else args.output
        img.save(path)
        saved.append(path)
 
    if args.num_images > 1:
        results.images[0].save(args.output)
 
    print(f"\n✓ Saved:")
    for p in saved:
        print(f"   {p}")
 
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()