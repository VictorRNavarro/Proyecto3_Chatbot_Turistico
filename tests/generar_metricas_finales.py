"""Genera la evidencia de evaluación solicitada para el Proyecto 3."""

import json
import sys
from pathlib import Path

import ollama

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import CACHE_DIR, EMBEDDING_MODEL, MAX_TURNS, MODELS_DIR, OLLAMA_MODEL
from src.chatbot_engine import TourismChatbot
from src.finetuning_utils import load_classifier
from src.rag_utils import RAGStore


PRUEBAS_RAG = [
    "Que hoteles tienen buena atencion al cliente?",
    "Dame otro del mismo tipo",
    "Y ese donde queda?",
    "Que restaurantes destacan por su comida?",
    "Recomiendame un parque con senderos bonitos",
    "Que opinan de los museos en San Jose?",
    "De que se quejan en los mercados artesanales?",
    "Que experiencias recomiendan en los tours de aventura?",
    "Que diferencia una reseña de un parque de una de un hotel?",
    "Cuanto cuesta un vuelo a Madrid?",
]

PRUEBAS_SIN_RAG = [PRUEBAS_RAG[0], PRUEBAS_RAG[4], PRUEBAS_RAG[6]]


def respuesta_sin_rag(pregunta: str) -> str:
    prompt = (
        "Eres un asistente turístico. Responde en español a la pregunta siguiente sin consultar "
        "reseñas ni una base de conocimiento. No afirmes que tu respuesta está respaldada por un corpus.\n\n"
        f"Pregunta: {pregunta}"
    )
    return ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])["message"]["content"]


def main() -> None:
    rag = RAGStore(CACHE_DIR, EMBEDDING_MODEL)
    classifier, classifier_model = load_classifier(MODELS_DIR / "clasificador_tipo_lugar")
    bot = TourismChatbot(rag, classifier, classifier_model, OLLAMA_MODEL, MAX_TURNS, top_k=4)

    conversaciones = []
    for numero, pregunta in enumerate(PRUEBAS_RAG, start=1):
        resultado = bot.respond(pregunta)
        conversaciones.append(
            {
                "id": numero,
                "modo": "con_rag",
                "pregunta": pregunta,
                "respuesta": resultado["answer"],
                "categoria_predicha": resultado["category"],
                "fuentes": [
                    {"lugar": fuente["lugar"], "tipo_lugar": fuente["tipo_lugar"], "score": fuente["score"]}
                    for fuente in resultado["sources"]
                ],
            }
        )
        print(f"RAG {numero}/10: {pregunta}")

    comparaciones = []
    for pregunta in PRUEBAS_SIN_RAG:
        con_rag = next(item for item in conversaciones if item["pregunta"] == pregunta)
        comparaciones.append(
            {
                "pregunta": pregunta,
                "respuesta_con_rag": con_rag["respuesta"],
                "fuentes_rag": con_rag["fuentes"],
                "respuesta_sin_rag": respuesta_sin_rag(pregunta),
            }
        )
        print(f"Baseline sin RAG: {pregunta}")

    metricas_clasificador = json.loads((ROOT / "resultados" / "metricas_clasificador.json").read_text(encoding="utf-8"))
    salida = {
        "proyecto": "Chatbot Turístico Inteligente",
        "modelo_generador": OLLAMA_MODEL,
        "clasificador": metricas_clasificador,
        "conversaciones": conversaciones,
        "comparaciones_con_y_sin_rag": comparaciones,
        "nota_metodologica": "Las respuestas con RAG incluyen las fuentes recuperadas. Las respuestas sin RAG se generaron con el mismo modelo local sin contexto del corpus.",
    }
    ruta = ROOT / "resultados" / "metricas.json"
    ruta.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nArchivo generado: {ruta}")


if __name__ == "__main__":
    main()
