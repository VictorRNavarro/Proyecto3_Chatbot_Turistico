"""Configuración centralizada de la aplicación."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "embeddings_cache"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "resultados"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "5"))
