"""Punto de entrada de la aplicación Dash."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dash import Dash, Input, Output, State, ctx, dcc, html
from app.config import CACHE_DIR, EMBEDDING_MODEL, MAX_TURNS, MODELS_DIR, OLLAMA_MODEL, TOP_K
from src.chatbot_engine import TourismChatbot
from src.finetuning_utils import load_classifier
from src.rag_utils import RAGStore

rag = RAGStore(CACHE_DIR, EMBEDDING_MODEL)
classifier, classifier_model = load_classifier(MODELS_DIR / "clasificador_tipo_lugar")
bot = TourismChatbot(rag, classifier, classifier_model, OLLAMA_MODEL, MAX_TURNS, TOP_K)

app = Dash(__name__)
app.title = "TicoGuía"
app.index_string = """<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>body{margin:0;background:radial-gradient(circle at 10% 8%,#d9f3df 0,transparent 29%),radial-gradient(circle at 90% 85%,#cceff0 0,transparent 30%),linear-gradient(135deg,#f5f0df,#dcefe8);min-height:100vh}*{box-sizing:border-box}</style>
</head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""
app.layout = html.Div([
    html.Div([
        html.Div("🌿", style={"fontSize": 38, "marginRight": 12}),
        html.Div([
            html.H1("TicoGuía", style={"margin": 0, "fontSize": 32, "letterSpacing": "-0.5px"}),
            html.P("Descubre Costa Rica a través de reseñas reales", style={"margin": "4px 0 0", "opacity": 0.9}),
        ]),
        html.Div("RAG + IA local", style={"marginLeft": "auto", "background": "rgba(255,255,255,.18)", "padding": "7px 11px", "borderRadius": 20, "fontSize": 12}),
    ], style={"display": "flex", "alignItems": "center", "color": "white", "padding": "24px 28px", "background": "linear-gradient(125deg, #075e54, #0a8967 58%, #1ba7a0)", "borderRadius": "22px 22px 0 0"}),
    html.Div([
        dcc.Store(id="messages", data=[]),
        html.Div([html.Span("✦ "), "Pregunta por hoteles, parques, restaurantes y experiencias turísticas."], style={"color": "#276749", "background": "#e9f8ef", "padding": "10px 14px", "borderRadius": 10, "fontSize": 14, "marginBottom": 14}),
        html.Div(id="chat", style={"height": "440px", "overflowY": "auto", "padding": "4px 6px 4px 0"}),
        html.Div([
            dcc.Textarea(id="question", placeholder="Ej.: ¿Qué hotel tiene buena atención al cliente?", style={"width": "100%", "height": 76, "padding": 13, "border": "1px solid #b7d9ca", "borderRadius": 12, "fontFamily": "inherit", "fontSize": 15, "boxSizing": "border-box", "resize": "vertical"}),
            html.Button("Enviar  ➜", id="send", n_clicks=0, style={"marginTop": 10, "background": "#e87a32", "color": "white", "border": 0, "padding": "11px 20px", "borderRadius": 9, "fontWeight": "bold", "fontSize": 14, "cursor": "pointer", "boxShadow": "0 4px 10px rgba(232,122,50,.22)"}),
            html.Button("Nueva conversación", id="clear", n_clicks=0, style={"margin": "10px 0 0 8px", "background": "white", "color": "#087f5b", "border": "1px solid #87b9a7", "padding": "10px 14px", "borderRadius": 9, "fontWeight": "bold", "cursor": "pointer"}),
        ]),
    ], style={"padding": "24px 28px 28px", "background": "rgba(255,255,255,.96)", "borderRadius": "0 0 22px 22px", "boxShadow": "0 18px 50px rgba(0,70,55,.18)"}),
], style={"maxWidth": "900px", "margin": "42px auto", "fontFamily": "Segoe UI, Arial, sans-serif"})

@app.callback(Output("messages", "data"), Input("send", "n_clicks"), Input("clear", "n_clicks"), State("question", "value"), State("messages", "data"), prevent_initial_call=True)
def answer(_, __, question, messages):
    if ctx.triggered_id == "clear":
        bot.reset()
        return []
    if not question or not question.strip(): return messages
    result = bot.respond(question.strip())
    sources = ", ".join(f"{item['lugar']} ({item['fuente']})" for item in result['sources']) or "sin fuentes"
    return messages + [{"role": "Usuario", "text": question}, {"role": "TicoGuía", "text": result['answer'], "sources": sources}]

@app.callback(Output("chat", "children"), Input("messages", "data"))
def render(messages):
    if not messages:
        return html.Div([html.Div("🦥", style={"fontSize": 42}), html.B("¡Pura vida! ¿A dónde quieres ir?"), html.P("Buscaré reseñas reales para ayudarte a elegir.", style={"margin": "6px 0", "color": "#5c6f68"})], style={"textAlign": "center", "padding": "110px 20px", "color": "#1f5141"})
    cards = []
    for message in messages:
        is_bot = message["role"] == "TicoGuía"
        cards.append(html.Div([
            html.Div("🌿 TicoGuía" if is_bot else "🧳 Tú", style={"fontSize": 12, "fontWeight": "bold", "marginBottom": 5, "color": "#087f5b" if is_bot else "#8c4b1e"}),
            html.Div(message["text"], style={"whiteSpace": "pre-wrap", "lineHeight": 1.45}),
            html.Small(f"Fuentes: {message['sources']}", style={"display": "block", "marginTop": 9, "color": "#53716a"}) if "sources" in message else None,
        ], style={"padding": "13px 15px", "margin": "10px 0", "background": "#edf9f3" if is_bot else "#fff4e9", "borderLeft": "4px solid #1b9b75" if is_bot else "4px solid #e87a32", "borderRadius": 10}))
    return cards

if __name__ == "__main__":
    app.run(debug=True)
