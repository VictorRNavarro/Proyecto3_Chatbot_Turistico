"""Evalúa el modelo guardado sobre test y persiste su matriz de confusión."""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.finetuning_utils import load_classifier

SEED = 42


def main() -> None:
    corpus = pd.read_csv(ROOT / "data" / "corpus_resenas.csv").dropna(subset=["texto", "tipo_lugar"]).copy()
    etiquetas = sorted(corpus["tipo_lugar"].unique())
    label2id = {etiqueta: indice for indice, etiqueta in enumerate(etiquetas)}
    corpus["label"] = corpus["tipo_lugar"].map(label2id)
    _, temporal = train_test_split(corpus, test_size=0.30, stratify=corpus["label"], random_state=SEED)
    _, test = train_test_split(temporal, test_size=0.50, stratify=temporal["label"], random_state=SEED)

    classifier, model = load_classifier(ROOT / "models" / "clasificador_tipo_lugar")
    predicciones = classifier(test["texto"].astype(str).str.slice(0, 512).tolist(), truncation=True, max_length=256, batch_size=32)
    labels_predichos = []
    for prediccion in predicciones:
        etiqueta = prediccion[0]["label"] if isinstance(prediccion, list) else prediccion["label"]
        labels_predichos.append(int(etiqueta.split("_")[-1]) if etiqueta.startswith("LABEL_") else label2id[etiqueta])

    matriz = confusion_matrix(test["label"], labels_predichos, labels=list(range(len(etiquetas)))).tolist()
    ruta_clasificador = ROOT / "resultados" / "metricas_clasificador.json"
    metricas_clasificador = json.loads(ruta_clasificador.read_text(encoding="utf-8"))
    metricas_clasificador["matriz_confusion"] = matriz
    ruta_clasificador.write_text(json.dumps(metricas_clasificador, ensure_ascii=False, indent=2), encoding="utf-8")

    ruta_final = ROOT / "resultados" / "metricas.json"
    metricas_finales = json.loads(ruta_final.read_text(encoding="utf-8"))
    metricas_finales["clasificador"] = metricas_clasificador
    ruta_final.write_text(json.dumps(metricas_finales, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Matriz {len(etiquetas)}x{len(etiquetas)} calculada sobre {len(test)} reseñas.")


if __name__ == "__main__":
    main()
