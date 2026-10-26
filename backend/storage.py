import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image, ImageOps

from backend.config import (
    ORIGINALS_DIR,
    VERSIONS_DIR,
    THUMBNAILS_DIR,
    METADATA_FILE,
    BASE_DIR
)


def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def create_thumbnail(image_path: Path, thumb_path: Path, size=(320, 320)) -> Path:
    """Create a high-quality centered thumbnail for fast gallery rendering."""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail(size, Image.Resampling.LANCZOS)
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(thumb_path, "JPEG", quality=85, optimize=True)
    return thumb_path


class StorageManager:
    """Manages image persistence, metadata catalog, and version history tree."""

    def __init__(self, metadata_path: Path = METADATA_FILE):
        self.metadata_path = metadata_path
        self._ensure_initialized()

    def _ensure_initialized(self):
        if not self.metadata_path.exists():
            self._save_data({"images": {}})

    def _load_data(self) -> Dict[str, Any]:
        try:
            if not self.metadata_path.exists():
                return {"images": {}}
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning loading metadata: {e}")
            return {"images": {}}

    def _save_data(self, data: Dict[str, Any]) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.metadata_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_file.replace(self.metadata_path)

    def get_all_images(self) -> List[Dict[str, Any]]:
        """Return list of all image records sorted by latest update."""
        data = self._load_data()
        images = list(data.get("images", {}).values())
        images.sort(key=lambda x: x.get("updated_at", x.get("created_at", "")), reverse=True)
        return images

    def get_image(self, image_id: str) -> Optional[Dict[str, Any]]:
        data = self._load_data()
        return data.get("images", {}).get(image_id)

    def add_original_image(
        self,
        pil_image: Image.Image,
        filename: str,
        title: Optional[str] = None,
        caption: Optional[str] = None,
        tags: Optional[List[str]] = None,
        objects: Optional[List[str]] = None,
        dominant_colors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Save a new original image, generate thumbnail, and register v0."""
        image_id = f"img_{uuid.uuid4().hex[:10]}"
        safe_suffix = Path(filename).suffix.lower()
        if safe_suffix not in [".jpg", ".jpeg", ".png", ".webp"]:
            safe_suffix = ".jpg"

        # Correct orientation
        pil_image = ImageOps.exif_transpose(pil_image)
        width, height = pil_image.size

        # File paths
        orig_filename = f"{image_id}_v0{safe_suffix}"
        orig_path = ORIGINALS_DIR / orig_filename
        thumb_filename = f"{image_id}_v0_thumb.jpg"
        thumb_path = THUMBNAILS_DIR / thumb_filename

        # Save original
        if safe_suffix in [".jpg", ".jpeg"]:
            pil_image.convert("RGB").save(orig_path, "JPEG", quality=95)
        else:
            pil_image.save(orig_path)

        # Generate thumbnail
        create_thumbnail(orig_path, thumb_path)

        now = _now_iso()
        v0_record = {
            "version_id": f"{image_id}_v0",
            "version_number": 0,
            "parent_version_id": None,
            "image_path": str(orig_path.relative_to(BASE_DIR)),
            "thumbnail_path": str(thumb_path.relative_to(BASE_DIR)),
            "timestamp": now,
            "edit_type": "original",
            "prompt": "Original Image",
            "model_used": "camera/upload",
            "notes": f"Original image {width}x{height}"
        }

        image_record = {
            "id": image_id,
            "title": title or Path(filename).stem.replace("_", " ").title(),
            "original_filename": filename,
            "created_at": now,
            "updated_at": now,
            "width": width,
            "height": height,
            "caption": caption or "Awaiting AI visual captioning...",
            "tags": tags or [],
            "objects": objects or [],
            "dominant_colors": dominant_colors or [],
            "embedding": None,
            "current_version_id": v0_record["version_id"],
            "versions": [v0_record]
        }

        data = self._load_data()
        data["images"][image_id] = image_record
        self._save_data(data)
        return image_record

    def update_image_metadata(
        self,
        image_id: str,
        caption: Optional[str] = None,
        tags: Optional[List[str]] = None,
        objects: Optional[List[str]] = None,
        dominant_colors: Optional[List[str]] = None,
        title: Optional[str] = None,
        embedding: Optional[List[float]] = None
    ) -> Optional[Dict[str, Any]]:
        """Update vision analysis, tags, title or vector embedding."""
        data = self._load_data()
        img = data.get("images", {}).get(image_id)
        if not img:
            return None

        if caption is not None:
            img["caption"] = caption
        if tags is not None:
            img["tags"] = tags
        if objects is not None:
            img["objects"] = objects
        if dominant_colors is not None:
            img["dominant_colors"] = dominant_colors
        if title is not None:
            img["title"] = title
        if embedding is not None:
            img["embedding"] = embedding

        img["updated_at"] = _now_iso()
        self._save_data(data)
        return img

    def add_version(
        self,
        image_id: str,
        edited_image: Image.Image,
        prompt: str,
        edit_type: str = "custom_prompt",
        model_used: str = "ai_api",
        notes: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Register a new edited version of an image, generating version file & thumbnail."""
        data = self._load_data()
        img = data.get("images", {}).get(image_id)
        if not img:
            return None

        current_ver_id = img["current_version_id"]
        version_num = len(img["versions"])
        new_version_id = f"{image_id}_v{version_num}"

        edited_image = ImageOps.exif_transpose(edited_image)
        new_filename = f"{new_version_id}.jpg"
        new_path = VERSIONS_DIR / new_filename
        thumb_filename = f"{new_version_id}_thumb.jpg"
        thumb_path = THUMBNAILS_DIR / thumb_filename

        edited_image.convert("RGB").save(new_path, "JPEG", quality=95)
        create_thumbnail(new_path, thumb_path)

        now = _now_iso()
        version_record = {
            "version_id": new_version_id,
            "version_number": version_num,
            "parent_version_id": current_ver_id,
            "image_path": str(new_path.relative_to(BASE_DIR)),
            "thumbnail_path": str(thumb_path.relative_to(BASE_DIR)),
            "timestamp": now,
            "edit_type": edit_type,
            "prompt": prompt,
            "model_used": model_used,
            "notes": notes or f"AI visual transformation v{version_num}"
        }

        img["versions"].append(version_record)
        img["current_version_id"] = new_version_id
        img["updated_at"] = now

        self._save_data(data)
        return img

    def revert_to_version(self, image_id: str, target_version_id: str) -> Optional[Dict[str, Any]]:
        """Change current active version pointer to a historical version."""
        data = self._load_data()
        img = data.get("images", {}).get(image_id)
        if not img:
            return None

        ver_exists = any(v["version_id"] == target_version_id for v in img["versions"])
        if not ver_exists:
            return None

        img["current_version_id"] = target_version_id
        img["updated_at"] = _now_iso()
        self._save_data(data)
        return img

    def delete_image(self, image_id: str) -> bool:
        """Completely remove an image record and its files from disk."""
        data = self._load_data()
        img = data.get("images", {}).pop(image_id, None)
        if not img:
            return False

        # Remove files safely
        for ver in img.get("versions", []):
            for path_key in ["image_path", "thumbnail_path"]:
                rel = ver.get(path_key)
                if rel:
                    file_path = BASE_DIR / rel
                    if file_path.exists():
                        try:
                            file_path.unlink()
                        except Exception as e:
                            print(f"Error removing {file_path}: {e}")

        self._save_data(data)
        return True

    def get_version_image_path(self, version_record: Dict[str, Any]) -> Path:
        """Resolve full filesystem Path from relative version path."""
        return BASE_DIR / version_record["image_path"]

    def get_active_version(self, image_record: Dict[str, Any]) -> Dict[str, Any]:
        """Get the version record corresponding to current_version_id."""
        cur_id = image_record.get("current_version_id")
        for v in image_record.get("versions", []):
            if v["version_id"] == cur_id:
                return v
        return image_record["versions"][-1]
