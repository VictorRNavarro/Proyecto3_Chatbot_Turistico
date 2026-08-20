# TicoGuía - Chatbot turístico con RAG

Proyecto 3 de Minería de Textos (CUC). Combina RAG sobre reseñas turísticas de Costa Rica, un clasificador fine-tuneado de tipo de lugar y una interfaz Dash.

## Organización del proyecto

```text
notebooks/   01 exploración, 02 RAG, 03 fine-tuning, 04 chatbot integrado
app/         interfaz Dash y configuración centralizada
src/         utilidades RAG, clasificación y motor conversacional
data/        corpus limpio y caché de embeddings/FAISS
models/      clasificador fine-tuneado guardado
resultados/  métricas, comparaciones y conversaciones de prueba
tests/       validaciones reproducibles del agente
```

## Instalación y ejecución

```powershell
pip install -r requirements.txt
ollama pull qwen3:1.7b
python app/chatbot_app.py
```

Abra `http://127.0.0.1:8050/`. Antes de usar la app, ejecute los notebooks 01 a 03 para generar el corpus, índice y modelo. El notebook 04 contiene pruebas conversacionales.

## Arquitectura

```text
Pregunta del usuario
        │
        ├─ Clasificador fine-tuneado → tipo de lugar
        │                                │
        └─ Embedding multilingüe → FAISS ─┘ → chunks filtrados
                                              │
Historial de los últimos 5 turnos ───────────┤
                                              ▼
                                 Prompt de TicoGuía + Qwen local
                                              │
                                              ▼
                                    Respuesta y fuentes en Dash
```

El notebook 02 compara chunking por oraciones y por párrafos; la estrategia final conserva una reseña por chunk y sus metadatos. Los embeddings y el índice FAISS se guardan en `data/embeddings_cache/`, de modo que no se regeneran al iniciar la aplicación.

### Configuración opcional

El modo predeterminado es local y no requiere API. Puede cambiarse sin editar código:

```powershell
$env:OLLAMA_MODEL = "qwen3:4b"  # si el equipo tiene 16 GB de RAM
$env:TOP_K = "4"
$env:MAX_TURNS = "5"
python app/chatbot_app.py
```

El notebook 02 incluye una celda independiente y opcional para comparar Gemini usando `GEMINI_API_KEY`; no afecta el funcionamiento local con Ollama.

## Resultados

El clasificador alcanzó accuracy de 80.08% y F1 macro de 75.07% sobre test; el baseline zero-shot obtuvo 62.67% y 51.98%, respectivamente.
