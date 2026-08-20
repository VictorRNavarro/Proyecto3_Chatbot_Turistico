"""Motor conversacional que integra clasificación, RAG, memoria y Ollama."""
from collections import deque
import ollama

SYSTEM_PROMPT = """Eres TicoGuía, un asesor turístico profesional especializado en Costa Rica.
Responde siempre en español, con un tono cordial, claro y práctico. Tu conocimiento proviene únicamente de las reseñas recuperadas que aparecen abajo.

Reglas:
- Recomienda solo lugares presentes en las reseñas recuperadas para la pregunta actual.
- Resume los aspectos concretos que mencionan las reseñas: atención, comida, limpieza, naturaleza, ubicación, precio u otros.
- No inventes direcciones, precios, horarios, servicios ni características que el contexto no confirme.
- Menciona únicamente atributos explícitos en las reseñas. Omite los aspectos sin evidencia; nunca los completes con frases como "es adecuado", "probablemente" o "no se menciona, pero".
- No deduzcas que un lugar es conveniente, recomendable o apto para una actividad si esa conclusión no aparece respaldada por una reseña recuperada.
- No menciones etiquetas, números de fuentes, chunks, modelos, RAG ni detalles técnicos. Las fuentes se muestran automáticamente en la interfaz.
- Para preguntas de seguimiento, usa el historial solo si ayuda a identificar el lugar o categoría solicitados.
- Si la evidencia no alcanza, responde exactamente: No tengo información suficiente en las reseñas recuperadas para responder esa pregunta.

Estructura sugerida: una respuesta directa; después, si aplica, una lista breve de recomendaciones con el motivo principal de cada una; cierra con una pregunta útil para continuar."""

MARCADORES_INFERENCIA = (
    "probablemente",
    "sugiere",
    "se deduce",
    "puede ser adecuado",
    "adecuado para",
    "es conveniente",
    "debería ser",
)


class TourismChatbot:
    def __init__(self, rag_store, classifier, classifier_model, ollama_model, max_turns=5, top_k=4):
        self.rag_store = rag_store
        self.classifier = classifier
        self.classifier_model = classifier_model
        self.ollama_model = ollama_model
        self.max_turns = max_turns
        self.top_k = top_k
        self.history = deque(maxlen=max_turns * 2)
        self.last_category = None
        self.last_sources = []
        self.last_query = None
        self.last_selected_place = None

    @staticmethod
    def _is_followup(question):
        text = question.lower()
        return any(fragment in text for fragment in ["otro del mismo", "otro similar", "y ese", "dónde queda", "donde queda", "háblame más", "hablame mas"])

    def respond(self, question):
        followup = self._is_followup(question)
        text = question.lower()
        category = self.last_category if followup and self.last_category else self._predict(question)
        alternative_followup = "otro" in text
        excluded = [item["lugar"] for item in self.last_sources] if alternative_followup else []
        location_followup = any(fragment in text for fragment in ["dónde queda", "donde queda", "y ese"])
        search_query = self.last_query if followup and self.last_query else question
        if location_followup and self.last_selected_place:
            results = self.rag_store.search_place(f"ubicación de {self.last_selected_place}", self.last_selected_place)
        else:
            results = self.rag_store.search(search_query, self.top_k, category, excluded)
        if alternative_followup and results:
            results = results[:1]
        if not results and followup and self.last_sources:
            results = self.last_sources
        context = "\n\n".join(
            f"Reseña {i} | {item['tipo_lugar']} | {item['lugar']} | {item['calificacion']} estrellas | {item['fuente']}: {item['texto']}"
            for i, item in enumerate(results, 1)
        )
        history = "\n".join(f"{role}: {message}" for role, message in self.history)
        prompt = f"{SYSTEM_PROMPT}\n\nHistorial:\n{history or '(sin historial)'}\n\nReseñas recuperadas:\n{context or '(sin evidencia)'}\n\nPregunta: {question}"
        if alternative_followup and results:
            item = results[0]
            answer = f"Como alternativa, {item['lugar']} ({item['calificacion']} estrellas). La reseña recuperada indica: {item['texto']}"
        elif location_followup and results:
            item = results[0]
            answer = f"Sobre {item['lugar']}, la reseña recuperada indica: {item['texto']}"
        elif not results:
            answer = "No tengo información suficiente en las reseñas recuperadas para responder esa pregunta."
        else:
            answer = ollama.chat(model=self.ollama_model, messages=[{"role": "user", "content": prompt}])["message"]["content"]
            # Algunos modelos locales continúan escribiendo después de reconocer
            # falta de evidencia. Se conserva solo la respuesta segura definida
            # por el asistente para evitar datos no respaldados por las reseñas.
            if "no tengo información suficiente" in answer.lower():
                answer = "No tengo información suficiente en las reseñas recuperadas para responder esa pregunta."
            else:
                # Se descartan líneas que transforman una reseña en una
                # inferencia. Los datos explícitos de las demás líneas se
                # mantienen disponibles para la persona usuaria.
                answer = "\n".join(
                    line for line in answer.splitlines()
                    if not any(marker in line.lower() for marker in MARCADORES_INFERENCIA)
                ).strip()
        self.history.extend([("Usuario", question), ("TicoGuía", answer)])
        self.last_category, self.last_sources = category, results
        if results:
            self.last_selected_place = results[0]["lugar"]
        if not followup:
            self.last_query = question
        return {"answer": answer, "category": category, "sources": results}

    def reset(self):
        self.history.clear()
        self.last_category = None
        self.last_sources = []
        self.last_query = None
        self.last_selected_place = None

    def _predict(self, question):
        result = self.classifier(question[:512])[0][0]["label"]
        if result.startswith("LABEL_"):
            return self.classifier_model.config.id2label[int(result.split("_")[-1])]
        return result
