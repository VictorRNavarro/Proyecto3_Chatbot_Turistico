"""Carga del RAG y búsqueda con metadatos turísticos."""
import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGStore:
    def __init__(self, cache_dir: Path, model_name: str):
        self.chunks = json.loads((cache_dir / "chunks.json").read_text(encoding="utf-8"))
        self.index = faiss.deserialize_index(np.frombuffer((cache_dir / "indice.faiss").read_bytes(), dtype=np.uint8))
        self.embedder = SentenceTransformer(model_name)

    def search(self, question, top_k=4, category=None, exclude_places=None, threshold=0.50):
        vector = self.embedder.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(vector)
        # Se amplía el conjunto de candidatos para poder ofrecer alternativas
        # cuando el usuario pide "otro del mismo tipo".
        scores, ids = self.index.search(vector, min(250, self.index.ntotal))
        excluded = {place.lower() for place in (exclude_places or [])}
        results, seen = [], set()
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or score < threshold:
                continue
            chunk = self.chunks[idx]
            if category and chunk["tipo_lugar"] != category:
                continue
            place = chunk["lugar"]
            if place.lower() in seen or place.lower() in excluded:
                continue
            seen.add(place.lower())
            results.append({**chunk, "score": float(score)})
            if len(results) == top_k:
                break
        return results

    def search_place(self, question, place, top_k=1):
        """Busca evidencia para un lugar ya seleccionado en la conversación."""
        vector = self.embedder.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(vector)
        scores, ids = self.index.search(vector, self.index.ntotal)
        candidates = []
        terms = ("alajuela", "centro", "aeropuerto", "san josé", "situad")
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if chunk["lugar"] != place:
                continue
            evidence = any(term in chunk["texto"].lower() for term in terms)
            candidates.append(({**chunk, "score": float(score)}, evidence))
        candidates.sort(key=lambda item: (item[1], item[0]["score"]), reverse=True)
        return [item[0] for item in candidates[:top_k]]
