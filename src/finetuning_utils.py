"""Carga del clasificador fine-tuneado."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def load_classifier(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=1), model


def predict_category(classifier, model, text):
    result = classifier(text[:512])[0][0]
    label = result["label"]
    if label.startswith("LABEL_"):
        return model.config.id2label[int(label.split("_")[-1])]
    return label
