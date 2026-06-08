import argparse
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', type=str, required=True, help='Text prompt')
    parser.add_argument('--output', type=str, required=True, help='Output filename')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stable Diffusion Text-to-Image Generator")
    print("=" * 60)
    
    print("\n[1/2] Loading Stable Diffusion 1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,  # Faster
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()  # VRAM optimization for 8GB
    print("      Model loaded (700MB)")
    
    print(f"\n[2/2] Generating image for: '{args.word}'")
    image = pipe(args.word).images[0]
    image.save(args.output)
    
    print(f"\n✓ Saved to {args.output}")
    print("=" * 60)

if __name__ == '__main__':
    main()