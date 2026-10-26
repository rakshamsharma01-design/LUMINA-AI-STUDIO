"""LUMINA AI - Next-Gen AI-Powered Image Editing Platform.

Frontend & Application Workflow built with Streamlit.
Supports Natural Language Search, AI Vision Captioning, Multi-Version Tracking,
Preset Transforms, and Generative Edits via OpenAI, Gemini, and Local Neural Engine.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import streamlit as st
from PIL import Image

# Backend imports
from backend.config import (
    BASE_DIR,
    PRESET_EDITS,
    OPENAI_API_KEY as DEFAULT_OPENAI_KEY,
    GEMINI_API_KEY as DEFAULT_GEMINI_KEY,
    OPENAI_API_BASE as DEFAULT_OPENAI_BASE,
    AZURE_OPENAI_MODEL as DEFAULT_AZURE_MODEL
)
from backend.storage import StorageManager
from backend.vision_captioner import analyze_image
from backend.edit_engine import EditEngine, synthesize_prompt_with_llm
from backend.search_engine import SearchEngine
from seed_sample_data import seed_library

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Lumina AI | Visual Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_path = BASE_DIR / "assets" / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State
if "storage" not in st.session_state:
    st.session_state.storage = StorageManager()
if "edit_engine" not in st.session_state:
    st.session_state.edit_engine = EditEngine()
if "search_engine" not in st.session_state:
    st.session_state.search_engine = SearchEngine()
if "selected_image_id" not in st.session_state:
    st.session_state.selected_image_id = None
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = DEFAULT_OPENAI_KEY
if "openai_api_base" not in st.session_state:
    st.session_state.openai_api_base = DEFAULT_OPENAI_BASE
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = DEFAULT_GEMINI_KEY
if "engine_mode" not in st.session_state:
    st.session_state.engine_mode = "openai" if DEFAULT_OPENAI_KEY else "local"
if "last_engineered_prompt" not in st.session_state:
    st.session_state.last_engineered_prompt = None

storage: StorageManager = st.session_state.storage
edit_engine: EditEngine = st.session_state.edit_engine
search_engine: SearchEngine = st.session_state.search_engine


# ==============================================================================
# Sidebar: Settings, Model Config & Library Status
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
      <span style="font-size:2rem;">✨</span>
      <div>
        <h2 style="margin:0; font-size:1.3rem; font-weight:800; background:linear-gradient(135deg, #6366f1, #ec4899); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">LUMINA AI</h2>
        <span style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em;">Generative Image Studio</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("⚙️ Engine & API Keys")

    is_azure = bool(st.session_state.openai_api_base and "azure" in st.session_state.openai_api_base.lower())
    key_label = "Azure OpenAI API Key" if is_azure else "OpenAI API Key"

    # API Keys Configuration
    user_openai_key = st.text_input(
        key_label,
        value=st.session_state.openai_api_key,
        type="password",
        help="Enables live Vision & prompt engineering synthesis (GPT-4.1-mini / GPT-4o)"
    )
    if user_openai_key != st.session_state.openai_api_key:
        st.session_state.openai_api_key = user_openai_key

    with st.expander("🌐 Custom API Endpoint / Azure Base URL", expanded=is_azure):
        user_endpoint = st.text_input(
            "API Base URL",
            value=st.session_state.openai_api_base,
            help="Azure AI Foundry endpoint (e.g. .../openai/v1) or custom base URL"
        )
        if user_endpoint != st.session_state.openai_api_base:
            st.session_state.openai_api_base = user_endpoint
            os.environ["OPENAI_API_BASE"] = user_endpoint

    user_gemini_key = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        help="Optional: Enables Gemini Flash Vision analysis & Imagen"
    )
    if user_gemini_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = user_gemini_key

    # Active Provider Selector
    available_engines = []
    if st.session_state.openai_api_key:
        engine_title = "🤖 Azure OpenAI (GPT-4.1-mini)" if is_azure else "🤖 OpenAI (GPT-4o + DALL-E 3)"
        available_engines.append(engine_title)
    if st.session_state.gemini_api_key:
        available_engines.append("🌟 Google Gemini (Flash Vision)")
    available_engines.append("⚡ Smart Local Engine (Zero Cost & Fast)")

    default_engine_idx = 0
    if st.session_state.engine_mode == "local" and "⚡ Smart Local Engine (Zero Cost & Fast)" in available_engines:
        default_engine_idx = available_engines.index("⚡ Smart Local Engine (Zero Cost & Fast)")
    elif st.session_state.engine_mode == "gemini":
        for idx, eng in enumerate(available_engines):
            if "Gemini" in eng:
                default_engine_idx = idx

    selected_engine_label = st.selectbox(
        "Active AI Engine",
        options=available_engines,
        index=default_engine_idx
    )
    if "OpenAI" in selected_engine_label or "Azure" in selected_engine_label:
        st.session_state.engine_mode = "openai"
    elif "Gemini" in selected_engine_label:
        st.session_state.engine_mode = "gemini"
    else:
        st.session_state.engine_mode = "local"

    # Status Pill
    if st.session_state.engine_mode == "openai":
        if is_azure:
            st.success("🟢 Azure GPT-4.1-mini Connected")
        else:
            st.success("🟢 OpenAI Cloud Connected")
    elif st.session_state.engine_mode == "gemini":
        st.success("🟢 Google Gemini Connected")
    else:
        st.info("⚡ Local Neural Engine Active (Offline Ready)")

    st.markdown("---")
    st.subheader("📊 Library Statistics")
    all_imgs = storage.get_all_images()
    total_versions = sum(len(img.get("versions", [])) for img in all_imgs)
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown(f"""
        <div class="stat-pill">
          <span class="label">Total Images</span>
          <span class="value">{len(all_imgs)}</span>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"""
        <div class="stat-pill">
          <span class="label">Total Versions</span>
          <span class="value">{total_versions}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Reset / Seed Sample Images", use_container_width=True):
        seed_library()
        st.success("Sample images refreshed!")
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#64748b; line-height:1.4;">
      <strong>Week 2 MVP Deliverable</strong><br>
      • Natural Language Image Search<br>
      • Prompt-driven Generative Edits<br>
      • Complete Version History Tree<br>
      • AI Vision Auto-Captioning
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# Main Navigation Header & Tabs
# ==============================================================================
st.markdown("""
<div class="lumina-header">
  <div class="lumina-title">
    <span>✨ LUMINA AI STUDIO</span>
  </div>
  <div class="lumina-subtitle">
    Next-generation visual editing platform powered by Vision & Generative AI APIs. Search in natural language, edit via prompts, and track version history.
  </div>
</div>
""", unsafe_allow_html=True)

nav_tabs = st.tabs([
    "🖼️ Image Library & Search",
    "🎨 Studio & Edit Mode",
    "📜 Version History & Lineage",
    "🧪 Prompt Engineering & Docs"
])


# ==============================================================================
# TAB 1: Image Library & Natural Language Search
# ==============================================================================
with nav_tabs[0]:
    all_images = storage.get_all_images()

    # Search Bar & Quick Concept Chips
    st.markdown("### 🔍 Search Image Collection in Natural Language")
    col_search, col_clear = st.columns([5, 1])
    with col_search:
        search_query = st.text_input(
            "Natural Language Search",
            placeholder="Try: 'show me the ones with a beach', 'photos with dogs', 'cozy coffee workspace', 'warm golden hour'...",
            label_visibility="collapsed"
        )
    with col_clear:
        if st.button("Clear Search", use_container_width=True):
            search_query = ""
            st.rerun()

    # Quick Search Chips
    chip_cols = st.columns(6)
    with chip_cols[0]:
        if st.button("🏖️ Beach Coast", use_container_width=True):
            search_query = "beach"
    with chip_cols[1]:
        if st.button("🐕 Pets & Dogs", use_container_width=True):
            search_query = "dog"
    with chip_cols[2]:
        if st.button("☕ Workspace", use_container_width=True):
            search_query = "coffee workspace"
    with chip_cols[3]:
        if st.button("👩 Portrait", use_container_width=True):
            search_query = "portrait woman"
    with chip_cols[4]:
        if st.button("🌅 Sunset / Warm", use_container_width=True):
            search_query = "golden hour sunset"
    with chip_cols[5]:
        if st.button("🎨 Edited Only", use_container_width=True):
            search_query = "edited"

    # Upload New Image Accordion
    with st.expander("📤 Upload New Image to Library", expanded=(len(all_images) == 0)):
        upload_col1, upload_col2 = st.columns([3, 2])
        with upload_col1:
            uploaded_file = st.file_uploader(
                "Choose an image (PNG, JPG, JPEG, WEBP)",
                type=["png", "jpg", "jpeg", "webp"],
                key="library_uploader"
            )
        with upload_col2:
            custom_title = st.text_input("Custom Title (Optional)", placeholder="e.g. Sunset in Malibu")
            auto_caption = st.checkbox("Auto-analyze image with AI Vision", value=True)

        if uploaded_file is not None:
            if st.button("🚀 Process & Add to Library", use_container_width=True):
                with st.spinner("Processing image and running AI visual analysis..."):
                    pil_img = Image.open(uploaded_file)
                    caption_data = {}
                    if auto_caption:
                        caption_data = analyze_image(
                            pil_img,
                            provider=st.session_state.engine_mode,
                            openai_key=st.session_state.openai_api_key,
                            gemini_key=st.session_state.gemini_api_key,
                            filename=uploaded_file.name,
                            base_url=st.session_state.get("openai_api_base")
                        )

                    new_record = storage.add_original_image(
                        pil_image=pil_img,
                        filename=uploaded_file.name,
                        title=custom_title or caption_data.get("title"),
                        caption=caption_data.get("caption"),
                        tags=caption_data.get("tags"),
                        objects=caption_data.get("objects"),
                        dominant_colors=caption_data.get("dominant_colors")
                    )
                    st.success(f"Added '{new_record['title']}' to your library!")
                    st.session_state.selected_image_id = new_record["id"]
                    st.rerun()

    st.markdown("---")

    # Perform Search
    if search_query:
        search_results = search_engine.search(
            search_query,
            all_images,
            openai_key=st.session_state.openai_api_key
        )
        st.markdown(f"**Found {len(search_results)} match{'es' if len(search_results) != 1 else ''} for:** *'{search_query}'*")
    else:
        search_results = [(img, 1.0, []) for img in all_images]
        st.markdown(f"**Showing all {len(all_images)} images in library**")

    # Gallery Grid
    if not search_results:
        st.info("No images matched your query. Try a different keyword like 'beach', 'sunset', 'portrait', or 'dog'.")
    else:
        cols_per_row = 3
        for i in range(0, len(search_results), cols_per_row):
            grid_cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(search_results):
                    img_rec, score, reasons = search_results[i + j]
                    active_ver = storage.get_active_version(img_rec)
                    thumb_path = BASE_DIR / active_ver["thumbnail_path"]
                    
                    with grid_cols[j]:
                        with st.container(border=True):
                            # Thumbnail image
                            if thumb_path.exists():
                                st.image(str(thumb_path), use_container_width=True)
                            else:
                                orig_full = BASE_DIR / active_ver["image_path"]
                                if orig_full.exists():
                                    st.image(str(orig_full), use_container_width=True)

                            # Title & Badges
                            ver_count = len(img_rec.get("versions", []))
                            active_ver_num = active_ver.get("version_number", 0)
                            
                            badge_html = f"<span class='badge-version'>v{active_ver_num} ({ver_count} revs)</span>"
                            if search_query and score > 0.1:
                                badge_html += f" <span class='badge-match'>{int(score*100)}% Match</span>"

                            st.markdown(f"""
                            <div style="margin-top:8px; margin-bottom:6px;">
                              <strong style="font-size:1.05rem; color:#f8fafc;">{img_rec.get('title', 'Untitled')}</strong><br>
                              {badge_html}
                            </div>
                            """, unsafe_allow_html=True)

                            # Search match reasons
                            if reasons:
                                reason_tags = " ".join([f"<span class='badge-tag'>{r}</span>" for r in reasons])
                                st.markdown(f"<div style='margin-bottom:6px;'>{reason_tags}</div>", unsafe_allow_html=True)

                            # Caption preview
                            caption_text = img_rec.get("caption", "")
                            if len(caption_text) > 110:
                                caption_text = caption_text[:110] + "..."
                            st.caption(caption_text)

                            # Tags
                            tags_html = " ".join([f"<span class='badge-tag'>#{t}</span>" for t in img_rec.get("tags", [])[:4]])
                            st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)

                            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                            btn_c1, btn_c2 = st.columns([3, 1])
                            with btn_c1:
                                if st.button(f"🎨 Edit in Studio", key=f"edit_btn_{img_rec['id']}", use_container_width=True):
                                    st.session_state.selected_image_id = img_rec["id"]
                                    st.success(f"Selected '{img_rec['title']}'! Switch to '🎨 Studio & Edit Mode' tab to start editing.")
                            with btn_c2:
                                if st.button("🗑️", key=f"del_btn_{img_rec['id']}", help="Delete image"):
                                    storage.delete_image(img_rec["id"])
                                    if st.session_state.selected_image_id == img_rec["id"]:
                                        st.session_state.selected_image_id = None
                                    st.rerun()


# ==============================================================================
# TAB 2: Studio & Edit Mode
# ==============================================================================
with nav_tabs[1]:
    all_images = storage.get_all_images()
    
    # If no image selected, provide selector
    if not all_images:
        st.warning("Your image library is empty. Please upload an image or click 'Reset / Seed Sample Images' in the sidebar.")
    else:
        # Image selector dropdown
        img_options = {img["id"]: f"{img['title']} (v{storage.get_active_version(img)['version_number']})" for img in all_images}
        default_index = 0
        if st.session_state.selected_image_id in img_options:
            default_index = list(img_options.keys()).index(st.session_state.selected_image_id)

        selected_id = st.selectbox(
            "Select Image to Edit",
            options=list(img_options.keys()),
            format_func=lambda x: img_options[x],
            index=default_index
        )
        st.session_state.selected_image_id = selected_id
        active_img_record = storage.get_image(selected_id)

        if active_img_record:
            active_version = storage.get_active_version(active_img_record)
            original_version = active_img_record["versions"][0]
            
            active_img_path = BASE_DIR / active_version["image_path"]
            orig_img_path = BASE_DIR / original_version["image_path"]

            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
              <div>
                <h3 style="margin:0; font-size:1.4rem;">{active_img_record['title']}</h3>
                <span style="color:#94a3b8; font-size:0.85rem;">
                  Active Version: <strong>v{active_version['version_number']}</strong> &nbsp;|&nbsp; 
                  Dimensions: <strong>{active_img_record['width']}x{active_img_record['height']}</strong> &nbsp;|&nbsp; 
                  Created: <strong>{active_img_record['created_at'][:10]}</strong>
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Two-column layout: Canvas Display (Left) + AI Edit Controls (Right)
            col_canvas, col_controls = st.columns([7, 5])

            with col_canvas:
                canvas_view_mode = st.radio(
                    "Canvas Display Mode",
                    ["Active Version", "Before & After (Original vs Active)"],
                    horizontal=True,
                    label_visibility="collapsed"
                )

                if canvas_view_mode == "Active Version":
                    if active_img_path.exists():
                        st.image(str(active_img_path), use_container_width=True, caption=f"Active Version: v{active_version['version_number']} — {active_version.get('prompt', '')}")
                else:
                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.markdown("<div style='text-align:center; font-weight:600; color:#94a3b8;'>ORIGINAL (v0)</div>", unsafe_allow_html=True)
                        if orig_img_path.exists():
                            st.image(str(orig_img_path), use_container_width=True)
                    with col_after:
                        st.markdown(f"<div style='text-align:center; font-weight:600; color:#818cf8;'>CURRENT (v{active_version['version_number']})</div>", unsafe_allow_html=True)
                        if active_img_path.exists():
                            st.image(str(active_img_path), use_container_width=True)

                # Metadata & AI Understanding Accordion
                with st.expander("👁️ AI Vision Understanding & Metadata Inspector", expanded=False):
                    st.markdown(f"**Caption:** {active_img_record.get('caption', 'N/A')}")
                    
                    obj_c, col_c = st.columns(2)
                    with obj_c:
                        st.markdown("**Detected Objects:**")
                        for obj in active_img_record.get("objects", []):
                            st.markdown(f"- {obj}")
                    with col_c:
                        st.markdown("**Dominant Colors:**")
                        for col in active_img_record.get("dominant_colors", []):
                            st.markdown(f"- {col}")

                    if active_img_record.get("tags"):
                        st.markdown("**Semantic Search Tags:**")
                        st.markdown(" ".join([f"`#{t}`" for t in active_img_record["tags"]]))

                # Download active version
                if active_img_path.exists():
                    with open(active_img_path, "rb") as f_img:
                        st.download_button(
                            label=f"💾 Download Active Image (v{active_version['version_number']})",
                            data=f_img.read(),
                            file_name=f"{active_img_record['id']}_v{active_version['version_number']}.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )

            with col_controls:
                st.markdown("### 🎛️ AI Edit Control Center")
                st.markdown(f"<div style='font-size:0.85rem; color:#94a3b8; margin-bottom:12px;'>Engine: <strong>{st.session_state.engine_mode.upper()}</strong></div>", unsafe_allow_html=True)

                # SECTION 1: 1-Click Curated Presets
                st.markdown("#### 1. Instant 1-Click Presets")
                preset_keys = list(PRESET_EDITS.keys())
                
                # Render preset buttons in 2 columns
                p_col1, p_col2 = st.columns(2)
                for idx, p_key in enumerate(preset_keys):
                    p_info = PRESET_EDITS[p_key]
                    target_col = p_col1 if idx % 2 == 0 else p_col2
                    with target_col:
                        if st.button(
                            f"{p_info['icon']} {p_info['label']}",
                            key=f"preset_btn_{p_key}",
                            use_container_width=True,
                            help=p_info['description']
                        ):
                            with st.spinner(f"Applying '{p_info['label']}'..."):
                                with Image.open(active_img_path) as current_pil:
                                    edited_pil, eng_prompt, model_name = edit_engine.apply_preset(
                                        current_pil,
                                        p_key,
                                        engine_mode=st.session_state.engine_mode,
                                        openai_key=st.session_state.openai_api_key,
                                        gemini_key=st.session_state.gemini_api_key
                                    )
                                    storage.add_version(
                                        image_id=active_img_record["id"],
                                        edited_image=edited_pil,
                                        prompt=p_info["label"],
                                        edit_type="preset",
                                        model_used=model_name,
                                        notes=f"Preset {p_info['label']}: {eng_prompt[:80]}"
                                    )
                                    st.session_state.last_engineered_prompt = eng_prompt
                                    st.success(f"Generated new version with {p_info['label']}!")
                                    st.rerun()

                st.markdown("---")

                # SECTION 2: Custom Natural Language Instruction
                st.markdown("#### 2. Custom Natural Language Instruction")
                st.caption("Instruct the AI in plain English to alter or augment the image.")

                # Prompt suggestion chips
                st.markdown("<span style='font-size:0.75rem; color:#94a3b8;'>Example instructions:</span>", unsafe_allow_html=True)
                sug_c1, sug_c2 = st.columns(2)
                with sug_c1:
                    if st.button("✂️ 'remove the background'", use_container_width=True):
                        st.session_state.custom_prompt_input = "remove the background and isolate subject on studio backdrop"
                    if st.button("🌳 'add trees in background'", use_container_width=True):
                        st.session_state.custom_prompt_input = "add lush green autumn trees in the background"
                with sug_c2:
                    if st.button("🌅 'change lighting to sunset'", use_container_width=True):
                        st.session_state.custom_prompt_input = "change colors and lighting to warm cinematic sunset golden hour"
                    if st.button("🕶️ 'give it cyberpunk neon glow'", use_container_width=True):
                        st.session_state.custom_prompt_input = "apply cyberpunk neon lighting with electric cyan and magenta reflections"

                user_instruction = st.text_area(
                    "Edit Prompt",
                    value=st.session_state.get("custom_prompt_input", ""),
                    placeholder="e.g. 'remove the person on the left', 'make it look like vintage 1970s film', 'add dramatic storm clouds'...",
                    height=90,
                    label_visibility="collapsed"
                )

                if st.button("🚀 Synthesize & Apply Edit", type="primary", use_container_width=True):
                    if not user_instruction.strip():
                        st.error("Please enter a natural language instruction.")
                    else:
                        with st.spinner("Step 1/2: Synthesizing photorealistic prompt engineering specification..."):
                            with Image.open(active_img_path) as current_pil:
                                edit_result = edit_engine.apply_custom_instruction(
                                    image=current_pil,
                                    instruction=user_instruction,
                                    image_caption=active_img_record.get("caption", ""),
                                    objects_list=active_img_record.get("objects", []),
                                    engine_mode=st.session_state.engine_mode,
                                    openai_key=st.session_state.openai_api_key,
                                    gemini_key=st.session_state.gemini_api_key
                                )
                                
                                storage.add_version(
                                    image_id=active_img_record["id"],
                                    edited_image=edit_result["edited_image"],
                                    prompt=user_instruction,
                                    edit_type="custom_prompt",
                                    model_used=edit_result["model_used"],
                                    notes=f"Category: {edit_result['edit_category']}"
                                )
                                st.session_state.last_engineered_prompt = edit_result["engineered_prompt"]
                                st.success("Edit successfully applied and registered as a new version!")
                                st.rerun()

                # Prompt Engineering Live Preview Box
                if st.session_state.last_engineered_prompt:
                    with st.expander("⚡ Prompt Engineering Translation Preview", expanded=True):
                        st.markdown(f"""
                        <div class="prompt-box">
                          <strong style="color:#a5b4fc;">Engineered Generation Prompt:</strong><br>
                          {st.session_state.last_engineered_prompt}
                        </div>
                        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 3: Version History & Lineage
# ==============================================================================
with nav_tabs[2]:
    all_images = storage.get_all_images()
    if not all_images:
        st.info("No images in library yet.")
    else:
        img_opts = {img["id"]: img["title"] for img in all_images}
        chosen_id = st.selectbox(
            "Select Image to View Version History",
            options=list(img_opts.keys()),
            format_func=lambda x: img_opts[x],
            index=list(img_opts.keys()).index(st.session_state.selected_image_id) if st.session_state.selected_image_id in img_opts else 0
        )
        img_record = storage.get_image(chosen_id)

        if img_record:
            st.markdown(f"### 📜 Version Lineage: *{img_record['title']}*")
            st.markdown(f"Total Revisions Recorded: **{len(img_record['versions'])}**")

            # Timeline Display
            for ver in reversed(img_record["versions"]):
                ver_id = ver["version_id"]
                is_active = (ver_id == img_record.get("current_version_id"))
                v_num = ver.get("version_number", 0)
                
                with st.container(border=True):
                    col_thumb, col_meta, col_actions = st.columns([2, 5, 2])
                    
                    with col_thumb:
                        thumb_file = BASE_DIR / ver["thumbnail_path"]
                        if thumb_file.exists():
                            st.image(str(thumb_file), use_container_width=True)
                        else:
                            full_file = BASE_DIR / ver["image_path"]
                            if full_file.exists():
                                st.image(str(full_file), use_container_width=True)

                    with col_meta:
                        badge_status = "<span class='badge-version' style='background:rgba(16,185,129,0.2); color:#34d399; border-color:#10b981;'>ACTIVE VERSION</span>" if is_active else ""
                        st.markdown(f"""
                        <div style="display:flex; align-items:center; gap:8px;">
                          <h4 style="margin:0;">Version {v_num}</h4>
                          {badge_status}
                        </div>
                        <div style="font-size:0.85rem; color:#94a3b8; margin-top:4px;">
                          <strong>Timestamp:</strong> {ver.get('timestamp', 'N/A')}<br>
                          <strong>Edit Type:</strong> <span class='badge-tag'>{ver.get('edit_type', 'N/A')}</span><br>
                          <strong>Model / Engine:</strong> {ver.get('model_used', 'N/A')}<br>
                          <strong>Prompt Instruction:</strong> <em>"{ver.get('prompt', 'N/A')}"</em>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_actions:
                        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                        if not is_active:
                            if st.button("⏪ Revert to this", key=f"revert_btn_{ver_id}", use_container_width=True):
                                storage.revert_to_version(img_record["id"], ver_id)
                                st.success(f"Reverted to Version {v_num}!")
                                st.rerun()

                        img_full_path = BASE_DIR / ver["image_path"]
                        if img_full_path.exists():
                            with open(img_full_path, "rb") as f_dl:
                                st.download_button(
                                    label="💾 Download",
                                    data=f_dl.read(),
                                    file_name=f"{img_record['id']}_v{v_num}.jpg",
                                    mime="image/jpeg",
                                    key=f"dl_ver_{ver_id}",
                                    use_container_width=True
                                )


# ==============================================================================
# TAB 4: Prompt Engineering & API Architecture Documentation
# ==============================================================================
with nav_tabs[3]:
    st.markdown("### 🧪 Prompt Engineering Studio & Architecture")

    st.markdown("""
    In modern AI-powered image editing, everyday creators express their creative intent in short, conversational instructions 
    like **'remove the background'**, **'add a tree in the background'**, or **'make it look like sunset'**.
    Directly feeding these short phrases to image generation APIs produces inconsistent results, hallucinations, or destroys subject identity.
    
    Our architecture uses a **Vision-Guided Two-Stage Prompt Engineering Pipeline**:
    """)

    st.markdown("""
    ```mermaid
    graph LR
        A[Original Image] --> B[Vision Analysis API / GPT-4o / Gemini]
        B --> C[Structured Scene Metadata: Subject, Lighting, Palette]
        D[User Casual Instruction] --> E[LLM Prompt Synthesizer]
        C --> E
        E --> F[Engineered Generation Specification]
        F --> G[Generative Image API / DALL-E 3 / Imagen / Local Engine]
        G --> H[Edited Image Version Cataloged]
    ```
    """)

    st.markdown("#### Live Prompt Synthesizer Sandbox")
    st.caption("Test how raw natural language is translated into production-grade prompts.")

    test_c1, test_c2 = st.columns(2)
    with test_c1:
        test_context = st.text_area(
            "Sample Image Context (from Vision API)",
            value="A smiling young woman in an autumn park with soft golden hour backlighting, trees, and bokeh.",
            height=100
        )
    with test_c2:
        test_raw_instruction = st.text_input(
            "User Raw Instruction",
            value="add a retro bicycle leaning on a park bench in the background"
        )
        if st.button("⚡ Test Prompt Synthesis", use_container_width=True):
            with st.spinner("Synthesizing prompt..."):
                synth_result = synthesize_prompt_with_llm(
                    user_instruction=test_raw_instruction,
                    image_caption=test_context,
                    openai_key=st.session_state.openai_api_key,
                    gemini_key=st.session_state.gemini_api_key
                )
                st.session_state.synth_demo = synth_result

    if "synth_demo" in st.session_state:
        res = st.session_state.synth_demo
        st.markdown(f"""
        <div class="glass-card">
          <strong style="color:#a5b4fc; font-size:1.1rem;">Synthesized Output</strong><br><br>
          <strong>Category:</strong> <span class="badge-tag">{res.get('edit_category', 'N/A')}</span><br>
          <strong>Summary:</strong> {res.get('summary_of_changes', 'N/A')}<br><br>
          <div class="prompt-box">
            <strong>Engineered Prompt:</strong><br>
            {res.get('engineered_prompt', 'N/A')}
          </div>
          <span style="font-size:0.8rem; color:#f43f5e;"><strong>Negative Constraints:</strong> {res.get('negative_prompt', 'None')}</span>
        </div>
        """, unsafe_allow_html=True)
