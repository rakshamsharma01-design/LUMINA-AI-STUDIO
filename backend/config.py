import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables if .env exists
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
ORIGINALS_DIR = IMAGES_DIR / "originals"
VERSIONS_DIR = IMAGES_DIR / "versions"
THUMBNAILS_DIR = IMAGES_DIR / "thumbnails"
METADATA_FILE = DATA_DIR / "metadata.json"
SAMPLES_DIR = BASE_DIR / "assets" / "samples"

# Ensure all directories exist
for p in [DATA_DIR, IMAGES_DIR, ORIGINALS_DIR, VERSIONS_DIR, THUMBNAILS_DIR, SAMPLES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# API Keys & Custom Endpoints
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
if AZURE_OPENAI_ENDPOINT.endswith("/responses"):
    AZURE_OPENAI_ENDPOINT = AZURE_OPENAI_ENDPOINT[:-len("/responses")]

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", AZURE_OPENAI_ENDPOINT).strip().rstrip("/")
if OPENAI_API_BASE.endswith("/responses"):
    OPENAI_API_BASE = OPENAI_API_BASE[:-len("/responses")]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or AZURE_OPENAI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()

# AI Models Configuration
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini").strip()
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", AZURE_OPENAI_MODEL).strip()
OPENAI_FAST_VISION_MODEL = os.getenv("OPENAI_FAST_VISION_MODEL", AZURE_OPENAI_MODEL).strip()
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "dall-e-3").strip()
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-flash-latest").strip()
GEMINI_PRO_MODEL = "gemini-flash-latest"
GEMINI_IMAGEN_MODEL = "imagen-3.0-generate-002"

# Curated Preset Edits
PRESET_EDITS = {
    "remove_background": {
        "label": "Cutout / Studio Clean",
        "description": "Isolate subject on a clean studio white/transparent aesthetic background",
        "category": "Cleanup",
        "icon": "✨",
        "prompt_template": "Isolate the main subject from the background, placing the subject against a clean, minimalist high-end studio photography background with soft ambient studio lighting and subtle ground shadow.",
        "local_filter": "studio_cutout"
    },
    "golden_hour": {
        "label": "Golden Hour Warmth",
        "description": "Infuse warm, radiant sunset rays, glowing highlights, and cinematic depth",
        "category": "Lighting",
        "icon": "🌅",
        "prompt_template": "Enhance lighting to cinematic golden hour sunset atmosphere. Add soft golden directional sunlight, warm amber rim light, subtle lens flare, and rich warm color tones while keeping subject details crisp.",
        "local_filter": "golden_hour"
    },
    "cyberpunk_neon": {
        "label": "Cyberpunk Neon Glow",
        "description": "Electrify the scene with vibrant magenta and cyan neon hues and dark moody tones",
        "category": "Style",
        "icon": "🌆",
        "prompt_template": "Transform the scene into a moody cyberpunk aesthetic with vibrant neon lights in electric cyan and vivid magenta, subtle rain-slick reflections, dark atmospheric contrast, and high-tech glow.",
        "local_filter": "cyberpunk"
    },
    "enhance_clarity": {
        "label": "Ultra Clarity & Contrast",
        "description": "Boost micro-contrast, sharpen details, and balance dynamic dynamic range",
        "category": "Enhance",
        "icon": "💎",
        "prompt_template": "Professional color grading and detail remastering. Enhance sharpness, micro-contrast, dynamic range in shadows and highlights, vibrant natural color balance, ultra high definition fidelity.",
        "local_filter": "enhance_clarity"
    },
    "vintage_film": {
        "label": "Vintage 1970s Film",
        "description": "Analog 35mm film grain, warm nostalgic kodachrome saturation, and soft vignetting",
        "category": "Style",
        "icon": "🎞️",
        "prompt_template": "Authentic vintage 1970s 35mm color film look. Add warm nostalgic Kodachrome color grading, subtle organic film grain, gentle fade in the deep blacks, and a delicate edge vignette.",
        "local_filter": "vintage_film"
    },
    "noir_monochrome": {
        "label": "Dramatic Noir Black & White",
        "description": "Deep rich blacks, crisp silvery highlights, and dramatic chiaroscuro contrast",
        "category": "Style",
        "icon": "🖤",
        "prompt_template": "Fine-art black and white photography with high tonal range. Deep velvety shadows, brilliant silver highlights, razor-sharp focus on texture and silhouette, dramatic chiaroscuro mood.",
        "local_filter": "noir"
    },
    "watercolor_art": {
        "label": "Ethereal Watercolor",
        "description": "Transform into delicate pastel watercolor painting with soft paper texture",
        "category": "Artistic",
        "icon": "🎨",
        "prompt_template": "Convert image into an expressive watercolor and ink fine-art illustration. Soft watercolor pigment bleeds, delicate paper texture, dreamy pastel palette, hand-painted aesthetic.",
        "local_filter": "watercolor"
    }
}
