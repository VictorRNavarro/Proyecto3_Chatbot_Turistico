"""Prueba manual reproducible de memoria conversacional del chatbot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import CACHE_DIR, EMBEDDING_MODEL, MAX_TURNS, MODELS_DIR, OLLAMA_MODEL
from src.chatbot_engine import TourismChatbot
from src.finetuning_utils import load_classifier
from src.rag_utils import RAGStore


PREGUNTAS = [
    "Que hoteles tienen buena atencion al cliente?",
    "Dame otro del mismo tipo",
    "Y ese donde queda?",
]


def main() -> None:
    rag = RAGStore(CACHE_DIR, EMBEDDING_MODEL)
    classifier, classifier_model = load_classifier(MODELS_DIR / "clasificador_tipo_lugar")
    bot = TourismChatbot(rag, classifier, classifier_model, OLLAMA_MODEL, MAX_TURNS, top_k=4)
    resultados = []
    for turno, pregunta in enumerate(PREGUNTAS, start=1):
        resultado = bot.respond(pregunta)
        resultados.append(resultado)
        lugares = [fuente["lugar"] for fuente in resultado["sources"]]
        print(f"\nTURNO {turno}: {pregunta}")
        print(f"Categoria: {resultado['category']}")
        print(f"Fuentes: {lugares}")
        print(f"Respuesta: {resultado['answer']}")

    fuentes_iniciales = {fuente["lugar"] for fuente in resultados[0]["sources"]}
    fuentes_alternativas = {fuente["lugar"] for fuente in resultados[1]["sources"]}
    fuentes_referidas = {fuente["lugar"] for fuente in resultados[2]["sources"]}
    assert resultados[1]["category"] == resultados[0]["category"], "No conservó la categoría"
    assert fuentes_alternativas.isdisjoint(fuentes_iniciales), "Repitió las fuentes al pedir otro"
    assert fuentes_referidas == fuentes_alternativas, "No conservó el referente de 'ese'"
    print("\nVALIDACIÓN APROBADA: categoría, alternativa y referente conservados.")


if __name__ == "__main__":
    main()
