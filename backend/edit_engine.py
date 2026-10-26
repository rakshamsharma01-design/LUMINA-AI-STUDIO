"""AI Edit Engine.

Executes generative image edits and transformations using OpenAI (DALL-E 3 / Edits),
Google Gemini/Imagen 3, and Local Smart Visual Processing.
"""

import os
import json
import base64
from io import BytesIO
from typing import Dict, Any, Optional, Tuple
import requests
from PIL import Image, ImageOps

from backend.config import (
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    OPENAI_IMAGE_MODEL,
    PRESET_EDITS,
    OPENAI_API_BASE,
    OPENAI_FAST_VISION_MODEL,
    GEMINI_VISION_MODEL
)
from backend.prompt_templates import (
    EDIT_PROMPT_SYNTHESIZER_SYSTEM,
    build_edit_synthesis_prompt,
    PRESET_PROMPT_ENGINEERED
)
from backend.local_editor import (
    apply_preset_filter,
    apply_natural_language_edit_local
)


def synthesize_prompt_with_llm(
    user_instruction: str,
    image_caption: str,
    objects_list: list = None,
    openai_key: Optional[str] = None,
    gemini_key: Optional[str] = None
) -> Dict[str, str]:
    """Expands casual user edit instructions into a prompt-engineered specification."""
    prompt_text = build_edit_synthesis_prompt(user_instruction, image_caption, objects_list)
    eff_openai_key = openai_key or os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    eff_gemini_key = gemini_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY

    # 1. Try OpenAI / Azure OpenAI if key available
    if eff_openai_key:
        try:
            eff_base = (os.getenv("OPENAI_API_BASE") or OPENAI_API_BASE or "https://api.openai.com/v1").rstrip("/")
            if eff_base.endswith("/responses"):
                eff_base = eff_base[:-len("/responses")]

            endpoint_url = f"{eff_base}/chat/completions"
            model = os.getenv("AZURE_OPENAI_MODEL") or OPENAI_FAST_VISION_MODEL or "gpt-4.1-mini"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {eff_openai_key}",
                "api-key": eff_openai_key
            }
            payload = {
                "model": model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": EDIT_PROMPT_SYNTHESIZER_SYSTEM},
                    {"role": "user", "content": prompt_text}
                ],
                "temperature": 0.4
            }
            resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                raw_json = data["choices"][0]["message"]["content"]
                return json.loads(raw_json)
        except Exception as e:
            print(f"Synthesizer OpenAI call failed, falling back: {e}")

    # 2. Try Gemini if key available
    if eff_gemini_key:
        try:
            gem_model = os.getenv("GEMINI_VISION_MODEL") or GEMINI_VISION_MODEL or "gemini-flash-latest"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gem_model}:generateContent?key={eff_gemini_key}"
            payload = {
                "contents": [
                    {"parts": [{"text": f"{EDIT_PROMPT_SYNTHESIZER_SYSTEM}\n\n{prompt_text}\nReturn JSON only."}]}
                ],
                "generationConfig": {"response_mime_type": "application/json", "temperature": 0.4}
            }
            resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
        except Exception as e:
            print(f"Synthesizer Gemini call failed, falling back: {e}")

    # 3. Rule-based prompt synthesizer fallback
    inst_lower = user_instruction.lower()
    engineered = (
        f"Professional photograph modification. Original scene context: {image_caption}. "
        f"Apply the following edit seamlessly: {user_instruction}. "
        f"Preserve original core subject identity and natural perspective lighting with photographic 8k fidelity."
    )
    category = "General Edit"
    if "remove background" in inst_lower:
        category = "Background Removal"
    elif "add" in inst_lower:
        category = "Object Insertion"
    elif "remove" in inst_lower:
        category = "Object Removal"
    elif any(k in inst_lower for k in ["color", "sunset", "lighting", "golden", "mood"]):
        category = "Color/Lighting"

    return {
        "engineered_prompt": engineered,
        "negative_prompt": "blurry, low quality, artifacts, distorted proportions, duplicate subjects",
        "edit_category": category,
        "summary_of_changes": f"Applied edit request: {user_instruction}"
    }


def generate_with_openai_dalle(
    prompt: str,
    api_key: Optional[str] = None
) -> Image.Image:
    """Generate or reimagine image via OpenAI DALL-E 3 API."""
    key = api_key or os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not key:
        raise ValueError("OpenAI API key missing")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    payload = {
        "model": "dall-e-3",
        "prompt": prompt[:1000],
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json"
    }

    resp = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    b64_data = resp.json()["data"][0]["b64_json"]
    return Image.open(BytesIO(base64.b64decode(b64_data)))


class EditEngine:
    """Dispatches edits to Cloud GenAI APIs or Local Fast Image Processing."""

    def __init__(self):
        pass

    def apply_preset(
        self,
        image: Image.Image,
        preset_key: str,
        engine_mode: str = "local",
        openai_key: Optional[str] = None,
        gemini_key: Optional[str] = None
    ) -> Tuple[Image.Image, str, str]:
        """Apply a curated preset edit.
        
        Returns:
            (edited_image, engineered_prompt, model_used)
        """
        preset_info = PRESET_EDITS.get(preset_key, {})
        engineered_prompt = PRESET_PROMPT_ENGINEERED.get(preset_key, preset_info.get("prompt_template", ""))

        if engine_mode == "openai" and openai_key:
            try:
                img = generate_with_openai_dalle(engineered_prompt, api_key=openai_key)
                return img, engineered_prompt, "OpenAI DALL-E 3"
            except Exception as e:
                print(f"OpenAI preset generation failed: {e}. Falling back to local engine.")

        # Local high fidelity engine
        local_result = apply_preset_filter(image, preset_key)
        return local_result, engineered_prompt, f"Smart Engine ({preset_info.get('label', preset_key)})"

    def apply_custom_instruction(
        self,
        image: Image.Image,
        instruction: str,
        image_caption: str = "",
        objects_list: list = None,
        engine_mode: str = "local",
        openai_key: Optional[str] = None,
        gemini_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process custom natural language edit instruction.
        
        Returns a dict containing:
            - edited_image: PIL Image
            - engineered_prompt: synthesized prompt
            - negative_prompt: negative constraints
            - edit_category: category classification
            - model_used: name of engine/model
            - summary: 1-sentence summary
        """
        # Step 1: Synthesize prompt
        synth = synthesize_prompt_with_llm(
            instruction,
            image_caption,
            objects_list,
            openai_key=openai_key,
            gemini_key=gemini_key
        )
        engineered_prompt = synth.get("engineered_prompt", instruction)

        # Step 2: Execute edit
        if engine_mode == "openai" and openai_key:
            try:
                gen_img = generate_with_openai_dalle(engineered_prompt, api_key=openai_key)
                return {
                    "edited_image": gen_img,
                    "engineered_prompt": engineered_prompt,
                    "negative_prompt": synth.get("negative_prompt", ""),
                    "edit_category": synth.get("edit_category", "Custom Edit"),
                    "model_used": "OpenAI DALL-E 3",
                    "summary": synth.get("summary_of_changes", instruction)
                }
            except Exception as e:
                print(f"OpenAI DALL-E 3 failed: {e}. Falling back to local engine.")

        # Local engine
        local_img = apply_natural_language_edit_local(image, instruction)
        return {
            "edited_image": local_img,
            "engineered_prompt": engineered_prompt,
            "negative_prompt": synth.get("negative_prompt", "blurry, low fidelity"),
            "edit_category": synth.get("edit_category", "Custom Edit"),
            "model_used": "Smart Neural Vision Engine (Local)",
            "summary": synth.get("summary_of_changes", instruction)
        }
