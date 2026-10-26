# ✨ LUMINA AI — Next-Gen AI Image Editing Platform (MVP)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/AI-OpenAI%20GPT--4o%20%7C%20DALL--E%203-412991.svg)](https://openai.com/)
[![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini%20%7C%20Imagen%203-4285F4.svg)](https://deepmind.google/technologies/gemini/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **MVP for an AI-First Visual Editing Platform** that empowers social media creators, marketers, product designers, and everyday users to search, view, and transform images using pure natural language — completely eliminating the steep learning curve of traditional legacy editing software.

---

## 🌟 Key Features

### 1. 🔍 Natural Language Image Search
- Search through your image collection using human queries such as:
  - *"Show me the ones with a beach"*
  - *"Photos with a dog in a park"*
  - *"Cozy coffee workspace with warm lighting"*
- **Hybrid Semantic & Vector Search**: Combines OpenAI embeddings (`text-embedding-3-small`), Gemini embeddings, and high-performance local TF-IDF / concept expansion so that queries match semantic meaning rather than just exact keywords.

### 2. 👁️ Multi-Modal Vision Auto-Captioning & Tagging
- When an image is uploaded, vision models (**OpenAI GPT-4o** or **Google Gemini 1.5 Flash**) automatically extract:
  - Evocative titles
  - Detailed photorealistic captions (subject pose, environment, lighting conditions)
  - 8–14 semantic tags
  - Detected objects and dominant color palettes
  - Suggested visual edits

### 3. 🎛️ AI Edit Control Center: 1-Click Presets & Custom Instructions
- **7 Curated 1-Click Presets**:
  - ✂️ **Studio Cutout**: Isolates subject on a clean studio neutral background with rim lighting.
  - 🌅 **Golden Hour Warmth**: Infuses 2800K radiant sunset rays, amber rim lights, and soft glow.
  - 🌆 **Cyberpunk Neon Glow**: Electrifies scene with cyan shadows, vivid magenta highlights, and neon glow.
  - 💎 **Ultra Clarity & Contrast**: Boosts micro-contrast, crisp textures, and balanced dynamic range.
  - 🎞️ **Vintage 1970s Film**: Analog 35mm film grain, nostalgic Kodachrome colors, and soft vignetting.
  - 🖤 **Dramatic Noir B&W**: Chiaroscuro black & white, deep velvety blacks, and brilliant highlights.
  - 🎨 **Ethereal Watercolor**: Fluid pastel watercolor washes with delicate ink contours.
- **Custom Natural Language Edit Instructions**:
  - Type plain-English requests like *"remove the background"*, *"add a tree in the background"*, or *"remove the person on the left"*.
  - **Vision-Guided Prompt Synthesizer**: Converts brief user instructions into prompt-engineered specifications preserving subject identity and scene lighting.

### 4. 📜 Non-Destructive Version History & Lineage Tree
- Complete visual timeline tracking every revision from `v0` (Original) to `v1`, `v2`, etc.
- Each version card preserves:
  - High-resolution thumbnail
  - Timestamp
  - Prompt / Preset used
  - Model / Engine tag
- **1-Click Rollback**: Revert to any historical version at any point without losing edits.
- **Before & After Comparison**: View the original and active version side-by-side.

### 5. ⚡ Hybrid Offline / Cloud Architecture
- **Cloud Ready**: Full support for OpenAI (GPT-4o + DALL-E 3) and Google Gemini (1.5 Flash + Imagen 3).
- **Offline / Zero-Cost Mode**: Built-in smart neural image processor using Pillow & NumPy ensures the application works **100% reliably out of the box** for presentations, portfolio demos, or evaluation without requiring immediate API keys.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Frontend [Streamlit Modern Glassmorphic UI]
        V1[🖼️ Image Library & NL Search]
        V2[🎨 Studio & Edit Mode]
        V3[📜 Version History & Lineage]
        V4[🧪 Prompt Playground & Docs]
    end

    subgraph Backend_Services [Lumina AI Backend Engine]
        SM[Storage Manager & Version Tree]
        VC[Vision Captioner: GPT-4o / Gemini / Heuristics]
        PE[Prompt Engineering Synthesizer]
        EE[Edit Engine: DALL-E 3 / Imagen / Local Neural]
        SE[Search Engine: Embeddings & TF-IDF]
    end

    subgraph Data_Store [Local File & Metadata Storage]
        IMG[(Originals, Versions, Thumbnails)]
        META[(metadata.json Catalog)]
    end

    V1 <--> SE
    V1 <--> SM
    V2 <--> VC
    V2 <--> PE
    V2 <--> EE
    V3 <--> SM
    SM <--> IMG
    SM <--> META
    SE <--> META
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Clone and Install Dependencies
```bash
git clone https://github.com/Moksh008/SIH26.git
cd "ai power image editing platform"

# Install dependencies
pip install -r requirements.txt
```

### 3. (Optional) Configure API Keys
You can set keys via `.env` file or directly inside the Streamlit app sidebar:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```
> *Note: If no API keys are provided, the platform automatically runs in Smart Local Engine mode with pre-seeded portfolio assets so you can immediately test all features!*

### 4. Launch the Platform
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```text
ai-power-image-editing-platform/
├── app.py                      # Main Streamlit Application (Library, Studio, Versions, Docs)
├── seed_sample_data.py         # Seeds realistic portfolio images and version history
├── requirements.txt            # Python dependencies
├── .env.example                # API key template
├── README.md                   # Complete platform documentation
├── assets/
│   ├── style.css               # Glassmorphic dark-theme custom stylesheet
│   └── samples/                # Seed sample images
├── backend/
│   ├── __init__.py
│   ├── config.py               # Paths, model configurations, and preset specs
│   ├── storage.py              # Version tracking, thumbnails, metadata persistence
│   ├── prompt_templates.py     # Prompt engineering templates for vision & generation
│   ├── vision_captioner.py     # Vision analysis (GPT-4o, Gemini, local heuristics)
│   ├── edit_engine.py          # AI image editing dispatcher & prompt synthesizer
│   ├── local_editor.py         # High-fidelity local image processing engine
│   └── search_engine.py        # Natural language semantic search (TF-IDF + embeddings)
├── data/
│   ├── images/                 # Image storage (originals, versions, thumbnails)
│   └── metadata.json           # Metadata catalog with version trees
├── docs/
│   └── API_AND_PROMPTS.md      # In-depth API selection & prompt engineering guide
└── notebooks/
    └── prompt_engineering.ipynb # Jupyter notebook demonstrating prompt handling
```

---

## 🎯 Week 2 Deliverables Alignment

| Objective / Deliverable | Status | Implementation Details |
| :--- | :---: | :--- |
| **Natural Language Editing Interface** | ✅ Completed | Streamlit Studio tab with 1-click presets & custom prompt input |
| **LLM + Vision Input Integration** | ✅ Completed | `backend/vision_captioner.py` and `backend/prompt_templates.py` |
| **Version History Logic** | ✅ Completed | `backend/storage.py` tracking parent-child revisions with thumbnails & timestamps |
| **Thumbnails & Rollback** | ✅ Completed | Auto-generated 320x320 thumbnails + 1-click version reversion |
| **Prompt Handling Scripts & Notebook** | ✅ Completed | `backend/prompt_templates.py` & `notebooks/prompt_engineering.ipynb` |
| **API Selection & Prompt Explanation** | ✅ Completed | `docs/API_AND_PROMPTS.md` |
| **Natural Language Search** | ✅ Completed | `backend/search_engine.py` matching queries like *"show me the ones with a beach"* |

---

## 🚢 Deployment Guide

### Deploying to Streamlit Community Cloud
1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select your repository, branch `main`, and main file `app.py`.
4. In Advanced Settings, add your `OPENAI_API_KEY` and/or `GEMINI_API_KEY` under Secrets.
5. Click **Deploy**.

### Deploying to Hugging Face Spaces
1. Create a new Space with the **Streamlit** SDK.
2. Push repository code to the Hugging Face Space repository.
3. Configure repository Secrets for API keys.
