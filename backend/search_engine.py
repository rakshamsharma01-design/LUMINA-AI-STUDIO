"""Natural Language Image Search Engine.

Supports:
1. OpenAI text-embedding-3-small semantic vector similarity
2. Google Gemini embeddings
3. Scikit-learn TF-IDF & Cosine Similarity local vector search
4. Keyword and Semantic Concept expansion matching
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import (
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    OPENAI_EMBEDDING_MODEL
)


# Semantic synonyms and plurals for common natural language visual queries
CONCEPT_MAP = {
    "beach": ["beach", "beaches", "ocean", "oceans", "sea", "seas", "sand", "sandy", "coast", "coastal", "shore", "palm", "palms", "tropical", "waves", "water"],
    "dog": ["dog", "dogs", "retriever", "retrievers", "puppy", "puppies", "canine", "canines", "pet", "pets", "animal", "animals", "fur", "golden", "hound"],
    "cat": ["cat", "cats", "kitten", "kittens", "feline", "felines", "pet", "pets", "whiskers"],
    "coffee": ["coffee", "coffees", "latte", "lattes", "espresso", "cup", "cups", "cappuccino", "cafe", "drink", "drinks", "table", "tables", "workspace", "workspaces", "desk"],
    "portrait": ["portrait", "portraits", "person", "persons", "people", "woman", "women", "man", "men", "girl", "girls", "face", "faces", "smile", "smiling", "lifestyle"],
    "sunset": ["sunset", "sunsets", "golden hour", "dusk", "evening", "sunlight", "warm", "amber", "glow"],
    "nature": ["nature", "trees", "tree", "park", "parks", "grass", "grassy", "leaves", "leaf", "autumn", "foliage", "outdoors", "outdoor", "greenery"],
    "edited": ["v1", "v2", "v3", "cyberpunk", "studio", "watercolor", "vintage", "noir", "edit", "edits", "edited", "preset", "reversion"]
}


def stem_word(w: str) -> str:
    """Lightweight morphological stemmer for English search tokens."""
    w = w.lower().strip()
    irregulars = {
        "leaves": "leaf", "leafs": "leaf", "women": "woman", "men": "man",
        "people": "person", "children": "child", "puppies": "puppy", "beaches": "beach"
    }
    if w in irregulars:
        return irregulars[w]
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4:
        if w.endswith(("shes", "ches", "sses", "xes", "zes")):
            return w[:-2]
        return w[:-1]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    return w


def get_openai_embedding(text: str, api_key: Optional[str] = None) -> Optional[List[float]]:
    """Compute vector embedding using OpenAI text-embedding-3-small."""
    key = api_key or os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not key:
        return None

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        payload = {
            "model": OPENAI_EMBEDDING_MODEL,
            "input": text.replace("\n", " ")
        }
        resp = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            return resp.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"OpenAI embedding error: {e}")
    return None


def get_gemini_embedding(text: str, api_key: Optional[str] = None) -> Optional[List[float]]:
    """Compute vector embedding using Google Gemini embedding model."""
    key = api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not key:
        return None

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={key}"
        payload = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text}]}
        }
        resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=20)
        if resp.status_code == 200:
            return resp.json()["embedding"]["values"]
    except Exception as e:
        print(f"Gemini embedding error: {e}")
    return None


class SearchEngine:
    """Hybrid semantic & concept search engine for image collections."""

    def __init__(self):
        pass

    def _prepare_document_text(self, img: Dict[str, Any]) -> str:
        """Create a rich searchable text representation of an image record, including stems."""
        raw_parts = [
            img.get("title", ""),
            img.get("caption", ""),
            " ".join(img.get("tags", [])),
            " ".join(img.get("objects", [])),
            " ".join(img.get("dominant_colors", [])),
            img.get("original_filename", "")
        ]
        # Include version prompts if available
        for v in img.get("versions", []):
            raw_parts.append(v.get("prompt", ""))
            raw_parts.append(v.get("edit_type", ""))
        
        combined = " ".join(raw_parts).lower()
        words = re_tokenize(combined)
        stems = [stem_word(w) for w in words]
        return combined + " " + " ".join(stems)

    def search(
        self,
        query: str,
        images: List[Dict[str, Any]],
        top_k: int = 24,
        openai_key: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], float, List[str]]]:
        """Perform natural language search over image library.
        
        Returns:
            List of tuples: (image_dict, relevance_score_0_to_1, matched_reasons)
        """
        if not images:
            return []

        clean_query = query.strip().lower()
        if not clean_query:
            # Return all images with neutral score 1.0
            return [(img, 1.0, ["Default order"]) for img in images]

        # 1. Clean conversational query prefixes repeatedly
        stripped_query = clean_query
        changed = True
        prefixes = [
            "show me the ones with a ", "show me the ones with ", "show me images of ",
            "show me pictures of ", "show me photos of ", "show me images with ",
            "show me photos with ", "show me ", "find pictures with ", "find pictures of ",
            "find photos with ", "find photos of ", "find images of ", "find ", "images with ",
            "photos with ", "pictures of ", "photos of ", "images of ", "pictures with ",
            "the ones with ", "a picture of ", "a photo of ", "an image of ", "pictures with a ",
            "photos with a ", "photos of a ", "pictures of a ", "images of a "
        ]
        while changed:
            changed = False
            for prefix in prefixes:
                if stripped_query.startswith(prefix):
                    stripped_query = stripped_query[len(prefix):].strip()
                    changed = True

        # 2. Vector search via TF-IDF (local, robust, instant)
        documents = [self._prepare_document_text(img) for img in images]
        
        # Add query and its stems to corpus
        query_tokens = re_tokenize(f"{clean_query} {stripped_query}")
        query_stems = [stem_word(w) for w in query_tokens]
        query_doc = f"{clean_query} {stripped_query} " + " ".join(query_stems)
        all_texts = documents + [query_doc]
        
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                token_pattern=r"(?u)\b\w+\b"
            )
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            doc_vectors = tfidf_matrix[:-1]
            query_vector = tfidf_matrix[-1:]
            
            cos_scores = cosine_similarity(query_vector, doc_vectors).flatten()
        except Exception as e:
            print(f"TF-IDF vectorizer error: {e}")
            cos_scores = np.zeros(len(images))

        # 3. Hybrid Semantic Concept Match & Tag Boosting
        STOP_WORDS = {
            "show", "me", "the", "ones", "with", "a", "an", "of", "in", "on", "at", "by", "for",
            "images", "image", "photos", "photo", "pictures", "picture", "find", "get", "look",
            "that", "have", "has", "had", "some", "all", "and", "or", "is", "are", "to", "from"
        }
        raw_words = [w for w in re_tokenize(clean_query + " " + stripped_query) if w not in STOP_WORDS]
        query_words = set(raw_words)
        
        # Expand query words with concept synonyms and stems
        expanded_concepts = set()
        for w in query_words:
            sw = stem_word(w)
            expanded_concepts.add(w)
            expanded_concepts.add(sw)
            for concept, synonyms in CONCEPT_MAP.items():
                synonym_stems = [stem_word(s) for s in synonyms]
                if w == concept or sw == concept or w in synonyms or sw in synonym_stems:
                    expanded_concepts.update(synonyms)
                    expanded_concepts.update(synonym_stems)

        results = []
        for idx, img in enumerate(images):
            tfidf_score = float(cos_scores[idx]) if idx < len(cos_scores) else 0.0
            
            matched_keywords = []
            bonus = 0.0
            
            title_lower = img.get("title", "").lower()
            caption_lower = img.get("caption", "").lower()
            tags_lower = [t.lower() for t in img.get("tags", [])]
            objects_lower = [o.lower() for o in img.get("objects", [])]

            title_tokens = set(re_tokenize(title_lower))
            title_stems = set(stem_word(t) for t in title_tokens)
            tag_tokens = set(re_tokenize(" ".join(tags_lower)))
            tag_stems = set(stem_word(t) for t in tag_tokens)
            object_tokens = set(re_tokenize(" ".join(objects_lower)))
            object_stems = set(stem_word(o) for o in object_tokens)
            caption_tokens = set(re_tokenize(caption_lower))
            caption_stems = set(stem_word(c) for c in caption_tokens)

            for term in expanded_concepts:
                if len(term) < 3:
                    continue
                st = stem_word(term)
                
                # Check title
                if term in title_tokens or st in title_stems or term in title_lower:
                    bonus += 0.35
                    matched_keywords.append(f"Title: '{term}'")
                
                # Check tags
                if term in tag_tokens or st in tag_stems or any(term in t for t in tags_lower):
                    bonus += 0.25
                    matched_keywords.append(f"Tag: #{term}")
                
                # Check objects
                if term in object_tokens or st in object_stems or any(term in o for o in objects_lower):
                    bonus += 0.20
                    matched_keywords.append(f"Object: {term}")
                
                # Check caption
                if term in caption_tokens or st in caption_stems or term in caption_lower:
                    bonus += 0.15
                    matched_keywords.append(f"Caption: '{term}'")

            # Final normalized composite score
            composite_score = min(1.0, (tfidf_score * 0.6) + min(0.6, bonus))

            # If user explicitly searched and image has positive match
            if composite_score > 0.05 or not clean_query:
                reasons = list(dict.fromkeys(matched_keywords))[:3]
                if not reasons and composite_score > 0.1:
                    reasons = [f"Semantic match ({int(composite_score*100)}%)"]
                results.append((img, round(composite_score, 3), reasons))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


def re_tokenize(text: str) -> List[str]:
    """Helper to split string into tokens without punctuation."""
    import re
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
