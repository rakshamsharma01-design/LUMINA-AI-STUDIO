"""High-quality local image processing engine using Pillow and NumPy.

Provides instantaneous, zero-cost, offline visual transformations for presets
and simulated natural language instructions.
"""

import math
from typing import Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw


def apply_preset_filter(image: Image.Image, preset_key: str) -> Image.Image:
    """Apply curated visual transformation matching the preset name."""
    image = ImageOps.exif_transpose(image).convert("RGB")

    if preset_key == "remove_background" or preset_key == "studio_cutout":
        return _apply_studio_cutout(image)
    elif preset_key == "golden_hour":
        return _apply_golden_hour(image)
    elif preset_key == "cyberpunk_neon" or preset_key == "cyberpunk":
        return _apply_cyberpunk(image)
    elif preset_key == "enhance_clarity":
        return _apply_enhance_clarity(image)
    elif preset_key == "vintage_film":
        return _apply_vintage_film(image)
    elif preset_key == "noir_monochrome" or preset_key == "noir":
        return _apply_noir_monochrome(image)
    elif preset_key == "watercolor_art" or preset_key == "watercolor":
        return _apply_watercolor(image)
    else:
        # Default subtle remaster
        return _apply_enhance_clarity(image)


def _apply_studio_cutout(image: Image.Image) -> Image.Image:
    """Simulate studio backdrop isolation with vignette gradient and subject focus."""
    w, h = image.size
    arr = np.array(image, dtype=np.float32)

    # Create smooth radial mask focusing on center subject
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    max_radius = math.sqrt(cx**2 + cy**2)
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2) / max_radius

    # Blur background portion
    blurred = np.array(image.filter(ImageFilter.GaussianBlur(radius=18)), dtype=np.float32)
    mask = np.clip((dist_from_center - 0.28) / 0.42, 0.0, 1.0)[:, :, np.newaxis]

    # Clean minimalist studio gradient
    studio_bg = np.zeros_like(arr)
    # Neutral cool grey studio gradient with gentle floor lighting
    gradient_vert = np.linspace(240, 215, h)[:, np.newaxis, np.newaxis]
    studio_bg[:] = gradient_vert

    # Blend original subject with softened studio backdrop
    blended = arr * (1.0 - mask) + (studio_bg * 0.75 + blurred * 0.25) * mask
    result = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    # Add gentle subject rim light
    enhancer = ImageEnhance.Sharpness(result)
    result = enhancer.enhance(1.25)
    return result


def _apply_golden_hour(image: Image.Image) -> Image.Image:
    """Infuse warm 2800K golden sunlight, amber highlights, and soft glow."""
    arr = np.array(image, dtype=np.float32)

    # Warm color grading: boost Red and slightly Green, gently suppress Blue in shadows
    arr[:, :, 0] = arr[:, :, 0] * 1.18 + 15.0  # Warm Red/Amber
    arr[:, :, 1] = arr[:, :, 1] * 1.06 + 8.0   # Soft Gold Green
    arr[:, :, 2] = arr[:, :, 2] * 0.88 - 5.0   # Warm shadow tint

    result = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Add soft warm glow bloom
    glow = result.filter(ImageFilter.GaussianBlur(radius=12))
    glow_arr = np.array(glow, dtype=np.float32)
    blended = np.clip(np.array(result, dtype=np.float32) * 0.82 + glow_arr * 0.22, 0, 255).astype(np.uint8)
    result = Image.fromarray(blended)

    # Enhance saturation & contrast
    result = ImageEnhance.Color(result).enhance(1.28)
    result = ImageEnhance.Contrast(result).enhance(1.12)
    return result


def _apply_cyberpunk(image: Image.Image) -> Image.Image:
    """Apply high contrast cyan shadows and vibrant magenta highlights with neon glow."""
    arr = np.array(image, dtype=np.float32)
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Shadows shift to deep electric cyan/teal
    shadow_mask = np.clip(1.0 - (lum / 140.0), 0.0, 1.0)[:, :, np.newaxis]
    # Highlights shift to intense magenta / neon violet
    highlight_mask = np.clip((lum - 110.0) / 145.0, 0.0, 1.0)[:, :, np.newaxis]

    # Cyan tint: low R, high G & B
    arr[:, :, 0] = arr[:, :, 0] * (1.0 - shadow_mask[:, :, 0] * 0.45)
    arr[:, :, 1] = arr[:, :, 1] * (1.0 + shadow_mask[:, :, 0] * 0.15)
    arr[:, :, 2] = arr[:, :, 2] * (1.0 + shadow_mask[:, :, 0] * 0.35)

    # Magenta tint: high R & B, lower G
    arr[:, :, 0] = arr[:, :, 0] * (1.0 + highlight_mask[:, :, 0] * 0.35)
    arr[:, :, 1] = arr[:, :, 1] * (1.0 - highlight_mask[:, :, 0] * 0.20)
    arr[:, :, 2] = arr[:, :, 2] * (1.0 + highlight_mask[:, :, 0] * 0.30)

    result = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    result = ImageEnhance.Contrast(result).enhance(1.35)
    result = ImageEnhance.Color(result).enhance(1.40)
    result = ImageEnhance.Sharpness(result).enhance(1.30)
    return result


def _apply_enhance_clarity(image: Image.Image) -> Image.Image:
    """Remaster clarity, micro-contrast, vibrant natural tones, and sharpness."""
    # Unsharp mask for high micro-contrast
    sharp = image.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3))
    enhanced = ImageEnhance.Contrast(sharp).enhance(1.16)
    enhanced = ImageEnhance.Color(enhanced).enhance(1.18)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.20)
    return enhanced


def _apply_vintage_film(image: Image.Image) -> Image.Image:
    """Nostalgic 1970s Kodachrome film look with warm matte tones and grain."""
    arr = np.array(image, dtype=np.float32)

    # Matte shadows: lift darkest blacks to warm charcoal
    arr = arr * 0.88 + 22.0

    # Kodachrome warm shift: boost yellow/red in highlights
    arr[:, :, 0] = arr[:, :, 0] * 1.08 + 6.0
    arr[:, :, 1] = arr[:, :, 1] * 1.02 + 4.0
    arr[:, :, 2] = arr[:, :, 2] * 0.90 - 4.0

    # Add subtle film grain
    grain = np.random.normal(0, 6.0, arr.shape)
    arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
    result = Image.fromarray(arr)

    # Subtle perimeter vignette
    w, h = result.size
    vignette = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(vignette)
    for r in range(min(w, h) // 2, 0, -10):
        alpha = int(255 * (r / (min(w, h) / 2)))
        draw.ellipse([(w//2 - r, h//2 - r), (w//2 + r, h//2 + r)], fill=alpha)
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=25))

    vignette_arr = np.array(vignette, dtype=np.float32) / 255.0
    vignette_arr = np.clip(vignette_arr * 0.35 + 0.65, 0.0, 1.0)[:, :, np.newaxis]

    final_arr = (np.array(result, dtype=np.float32) * vignette_arr).astype(np.uint8)
    return Image.fromarray(final_arr)


def _apply_noir_monochrome(image: Image.Image) -> Image.Image:
    """High-contrast black & white with dramatic chiaroscuro and crisp silver highlights."""
    gray = ImageOps.grayscale(image)
    # High contrast curve
    contrasted = ImageEnhance.Contrast(gray).enhance(1.45)
    sharp = ImageEnhance.Sharpness(contrasted).enhance(1.30)
    return sharp.convert("RGB")


def _apply_watercolor(image: Image.Image) -> Image.Image:
    """Artistic watercolor painting with soft pigment wash and delicate contours."""
    # Smooth base
    smooth = image.filter(ImageFilter.MedianFilter(size=5))
    smooth = smooth.filter(ImageFilter.SMOOTH_MORE)

    # Subtle edge lines
    edges = image.filter(ImageFilter.FIND_EDGES).convert("L")
    edges = ImageOps.invert(edges).point(lambda p: 255 if p > 220 else p)
    edge_arr = np.array(edges.convert("RGB"), dtype=np.float32) / 255.0

    smooth_arr = np.array(smooth, dtype=np.float32)
    painted = np.clip(smooth_arr * (edge_arr * 0.4 + 0.6), 0, 255).astype(np.uint8)

    result = Image.fromarray(painted)
    result = ImageEnhance.Color(result).enhance(1.32)
    return result


def apply_natural_language_edit_local(image: Image.Image, instruction: str) -> Image.Image:
    """Intelligently interpret natural language instructions and apply appropriate visual edits."""
    inst_lower = instruction.lower().strip()

    # Background removal / isolation
    if any(k in inst_lower for k in ["remove background", "cut out", "isolate", "studio background", "transparent", "white background"]):
        return _apply_studio_cutout(image)

    # Lighting / sunset / golden hour
    if any(k in inst_lower for k in ["sunset", "golden hour", "warm", "sunlight", "golden", "amber", "morning sun"]):
        return _apply_golden_hour(image)

    # Cyberpunk / neon / futuristic
    if any(k in inst_lower for k in ["cyberpunk", "neon", "glow", "magenta", "futuristic", "sci-fi", "electric"]):
        return _apply_cyberpunk(image)

    # Black and white / monochrome / noir
    if any(k in inst_lower for k in ["black and white", "monochrome", "noir", "grayscale", "b&w", "remove color"]):
        return _apply_noir_monochrome(image)

    # Vintage / retro / 70s / film
    if any(k in inst_lower for k in ["vintage", "retro", "film", "1970", "analog", "kodak", "polaroid", "nostalgic"]):
        return _apply_vintage_film(image)

    # Watercolor / painting / artistic
    if any(k in inst_lower for k in ["watercolor", "painting", "artistic", "sketch", "draw", "canvas", "illustration"]):
        return _apply_watercolor(image)

    # Brighten / lighten
    if any(k in inst_lower for k in ["bright", "lighten", "expose", "sunny"]):
        res = ImageEnhance.Brightness(image).enhance(1.25)
        return ImageEnhance.Contrast(res).enhance(1.10)

    # Darken / moody / night
    if any(k in inst_lower for k in ["dark", "moody", "night", "dim"]):
        res = ImageEnhance.Brightness(image).enhance(0.78)
        return ImageEnhance.Contrast(res).enhance(1.20)

    # Sharpen / enhance / details / hdr
    if any(k in inst_lower for k in ["sharp", "enhance", "clarity", "hd", "4k", "detail", "hdr", "clear"]):
        return _apply_enhance_clarity(image)

    # Vibrant / saturated / colorful
    if any(k in inst_lower for k in ["vibrant", "colorful", "saturation", "pop", "color"]):
        res = ImageEnhance.Color(image).enhance(1.45)
        return ImageEnhance.Contrast(res).enhance(1.12)

    # Object addition / removal simulation: high fidelity focal remaster
    return _apply_enhance_clarity(image)
