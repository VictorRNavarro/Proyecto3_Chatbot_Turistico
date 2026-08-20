# Uso de IA - Proyecto 3: Chatbot Turístico con RAG y Fine-Tuning

## Herramientas utilizadas

- **Codex (OpenAI):** apoyo para revisar la rúbrica, organizar y depurar el pipeline RAG, el clasificador fine-tuneado, la memoria conversacional y la interfaz en Plotly Dash.
- **Qwen3 1.7B mediante Ollama:** modelo local utilizado por el chatbot para generar respuestas a partir de las reseñas recuperadas. Se mantuvo como modo principal para que el proyecto funcione sin una API de pago.
- **Gemini:** alternativa opcional de prueba en el notebook 02; no es necesaria para ejecutar la aplicación ni para la demostración.


## Ejemplos de prompts utilizados

1. "En base al documento del proyecto, revisa qué requisitos faltan en los notebooks de RAG, fine-tuning y chatbot."
2. "Hay que disminuir la cantidad mínima de palabras para que el corpus tenga más de 5,000 datos."
3. "Creo que deberíamos mejorar el prompt del chatbot más profesional y sin usar [n]."
4. "Probemos validar la memoria del agente con 'Dame otro del mismo tipo' y '¿Y ese dónde queda?'."
5. "Actualiza el notebook 04 para que use el mismo motor del chatbot que la aplicación Dash."

## Validación de las sugerencias

La IA se utilizó como apoyo técnico, no como sustituto del análisis. Cada sugerencia se ejecutó y se revisó antes de aceptarse:

- Se ejecutaron los cuatro notebooks y se verificó que no tuvieran errores.
- Se comprobó que el corpus contuviera 5,018 reseñas, siete categorías y los campos requeridos.
- Se revisaron accuracy, F1 macro, reporte de clasificación y matriz de confusión sobre el conjunto de test.
- Se compararon respuestas con y sin RAG y se documentaron diez conversaciones en `resultados/metricas.json`.
- Se probaron memoria conversacional, alternativas, ubicación y preguntas fuera de dominio con Ollama local.

## Reflexión sobre el uso de IA

La IA ayudó a comprender técnicas de RAG, fine-tuning y agentes conversacionales, a evaluar alternativas de implementación y a detectar problemas de recuperación y memoria. Sin embargo, los modelos se ejecutaron sobre el corpus turístico del proyecto y los resultados se revisaron antes de interpretarlos.
