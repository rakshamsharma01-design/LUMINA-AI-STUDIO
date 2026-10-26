"""Prompt Engineering Templates for AI-Powered Image Editing Platform.

Includes:
1. Vision Analysis & Multi-Modal Captioning Templates (GPT-4o / Gemini Vision)
2. Natural Language Edit Translation Templates (translates casual user requests to photorealistic API prompts)
3. Specialized Category Templates (Background Removal, Object Addition/Removal, Color Grading, Style Transfer)
"""

# ==============================================================================
# 1. VISION ANALYSIS SYSTEM PROMPT
# ==============================================================================

VISION_ANALYSIS_SYSTEM_PROMPT = """You are an expert computer vision and image cataloging AI.
Analyze the provided image with high precision and provide a structured JSON response.

Your task:
1. Generate an evocative, professional title (3-5 words).
2. Generate a comprehensive, photorealistic visual caption (2-3 sentences) detailing:
   - The primary subject(s) and their pose/action.
   - Background environment, architecture, or nature.
   - Lighting conditions (e.g., golden hour, overcast, harsh direct, studio softbox).
   - Color palette and atmosphere.
3. Extract 8-14 semantic keywords/tags that users might search for (e.g., 'beach', 'sunset', 'woman', 'portrait', 'golden hour', 'autumn', 'coffee', 'vintage').
4. List specific prominent objects or entities visible.
5. Identify the dominant colors and mood.
6. Identify 2-3 potential edit suggestions that would visually complement or enhance the photo.

Respond ONLY with valid JSON conforming to this schema:
{
  "title": "string",
  "caption": "string",
  "tags": ["string", "string"],
  "objects": ["string", "string"],
  "dominant_colors": ["string", "string"],
  "mood": "string",
  "suggested_edits": ["string", "string"]
}
"""

VISION_ANALYSIS_USER_PROMPT = """Analyze this image thoroughly according to the schema. Ensure the caption and tags capture rich visual semantics so users can find this image using natural language queries like 'show me the ones with a beach' or 'photos with animals'."""


# ==============================================================================
# 2. EDIT INSTRUCTION EXPANSION & SYNTHESIZER
# ==============================================================================

EDIT_PROMPT_SYNTHESIZER_SYSTEM = """You are a senior Generative AI Prompt Engineer specializing in image-to-image synthesis, inpainting, and generative editing (OpenAI DALL-E 3, Gemini Imagen 3, Stable Diffusion).

When an everyday user enters a brief or casual edit instruction (e.g., "remove the background", "add a tree in the background", "remove the person on the left", "make it look like sunset"), your role is to translate that instruction into a rigorous, photorealistic image generation / editing prompt that preserves the identity and geometry of untouched areas while seamlessly integrating the requested changes.

Guidelines:
1. Subject Preservation: Clearly identify what must be kept intact (subject pose, expression, core silhouette, textures).
2. Lighting & Shadow Coherence: Specify that new elements or background changes must inherit the exact color temperature, shadows, and reflections of the scene.
3. Photorealism & High Fidelity: Avoid generic buzzwords; specify tangible optics (e.g., 35mm lens, natural depth of field, 8k resolution, raw photography look).
4. Unambiguous Edit Execution:
   - For "remove background": Specify isolating the main subject against a pristine minimalist backdrop or specific environment.
   - For "add X": Detail where X is placed, its scale, perspective, lighting alignment, and interaction with ground/air.
   - For "remove X": Describe inpainting and filling the vacated space with seamless surrounding texture, foliage, or background geometry.
   - For "change color/lighting": Detail precise tint, Kelvin temperature, contrast balance, and highlight roll-off.

Respond in JSON format:
{
  "engineered_prompt": "The complete, highly detailed prompt ready for image generation/editing API",
  "negative_prompt": "Undesired artifacts (e.g. blur, bad anatomy, duplicate subjects, artifacts)",
  "edit_category": "Background Removal | Object Insertion | Object Removal | Color/Lighting | Style Transfer",
  "summary_of_changes": "A concise 1-sentence explanation of what will be modified"
}
"""


def build_edit_synthesis_prompt(
    user_instruction: str,
    image_caption: str,
    objects_list: list = None
) -> str:
    """Builds the dynamic prompt sent to the LLM to expand casual edit instructions."""
    objs_str = ", ".join(objects_list) if objects_list else "Not explicitly parsed"
    return f"""IMAGE CONTEXT:
- Visual Description: {image_caption}
- Prominent Elements/Objects: {objs_str}

USER EDIT REQUEST:
"{user_instruction}"

TASK:
Synthesize an optimal Generative AI image editing prompt that faithfully applies the user's edit request to the existing image while preserving subject continuity and natural lighting. Return the JSON structure."""


# ==============================================================================
# 3. DIRECT DALL-E / IMAGEN EDIT TEMPLATES
# ==============================================================================

PRESET_PROMPT_ENGINEERED = {
    "remove_background": (
        "Professional commercial studio cutout. The primary subject from the original image "
        "is flawlessly extracted and isolated against a clean, modern studio background with a gentle "
        "neutral gradient, soft ambient floor shadow, crisp rim lighting, pristine edges, and studio photography lighting."
    ),
    "golden_hour": (
        "Remastered with cinematic golden hour sunset lighting. Warm, radiant 2800K amber sunlight sweeping "
        "across the scene from the side, delicate lens flare, golden rim lighting defining silhouettes, rich warm color tones, "
        "deep cinematic contrast, and natural atmospheric glow while preserving subject sharpness."
    ),
    "cyberpunk_neon": (
        "Reimagined as a striking neo-Tokyo cyberpunk scene. Dark moody atmosphere illuminated by electric cyan and "
        "vivid magenta neon lighting tubes, subtle reflections on damp asphalt surfaces, glowing holographic accents, high "
        "dynamic contrast, cinematic sci-fi tone, maintaining the original subject silhouette."
    ),
    "enhance_clarity": (
        "Masterwork ultra-high-definition photographic enhancement. Crisp micro-contrast, perfectly balanced dynamic "
        "range between highlights and shadows, true-to-life vibrant color grading, pristine texture definition, no digital noise, "
        "clean 8k medium-format aesthetic."
    ),
    "vintage_film": (
        "Authentic vintage 1970s analog 35mm film photography. Gentle organic film grain, warm nostalgic Kodachrome color "
        "palette, slightly faded deep blacks, delicate warm cream highlights, subtle chromatic aberration, and a soft perimeter vignette."
    ),
    "noir_monochrome": (
        "Dramatic black and white fine-art photography. High chiaroscuro contrast, deep velvety blacks, brilliant crisp silver "
        "highlights, razor-sharp edge definition, dramatic emotional shadow play, and rich mid-tone gradation."
    ),
    "watercolor_art": (
        "Expressive handmade watercolor and ink fine-art illustration. Soft pigment bleeding across textured cold-press watercolor "
        "paper, delicate ink outlines, pastel luminous washes, spontaneous brush splatters, fluid artistic composition."
    )
}


# ==============================================================================
# 4. NATURAL LANGUAGE SEARCH TEMPLATES
# ==============================================================================

SEARCH_REASONING_PROMPT = """You are an intelligent visual search assistant.
A user is searching an image library with this query: "{query}"

Given the following candidate images with their captions and tags:
{candidates_json}

Rank and select the most relevant images that match the user's intent. Consider synonyms, related concepts (e.g. 'beach' matches 'ocean', 'coast', 'waves', 'tropical sand'), and mood.
Return a JSON array of matching image IDs in order of relevance, with a brief explanation for each match.
"""
