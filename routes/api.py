import time
import json
from flask import Blueprint, request, jsonify, render_template

from preprocessing.text_cleaner import clean_text
from database.db import log_prediction, get_recent_predictions, get_stats

api_bp = Blueprint("api", __name__)


def load_model_globals():
    global vectorizer, classifier
    import os, joblib
    from config import MODELS_DIR
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    classifier = joblib.load(os.path.join(MODELS_DIR, "rf_classifier.pkl"))


vectorizer = None
classifier = None


def extract_indicators(text, feature_names, importances, threshold=0.02):
    keywords_phishing = [
        "urgent", "verify", "account", "login", "password", "click", "update",
        "confirm", "security", "suspend", "restrict", "credential", "bank",
        "payment", "invoice", "alert", "limited", "action required", "free",
        "winner", "claim", "prize", "lottery", "congratulation",
    ]
    text_lower = text.lower()
    found = []
    for kw in keywords_phishing:
        if kw in text_lower:
            found.append(f"{kw.capitalize()} language detected")
    return found[:6] if found else ["Standard email pattern"]


@api_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "email_text" not in data:
        return jsonify({"error": "Missing email_text field"}), 400

    email_text = str(data["email_text"]).strip()
    if not email_text:
        return jsonify({"error": "Email text cannot be empty"}), 400

    from config import MAX_TEXT_LENGTH
    if len(email_text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"Text exceeds {MAX_TEXT_LENGTH} characters"}), 400

    t0 = time.time()
    cleaned = clean_text(email_text)

    if vectorizer is None or classifier is None:
        load_model_globals()

    vec = vectorizer.transform([cleaned])
    pred = classifier.predict(vec)[0]
    proba = classifier.predict_proba(vec)[0]

    is_phishing = bool(pred == 1)
    confidence = float(max(proba) * 100)
    label = "Phishing" if is_phishing else "Legitimate"

    indicators = extract_indicators(email_text, [], [])

    processing_time_ms = round((time.time() - t0) * 1000, 2)

    log_prediction(email_text, label, confidence, is_phishing, processing_time_ms, indicators)

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 2),
        "is_phishing": is_phishing,
        "indicators": indicators,
        "processing_time_ms": processing_time_ms,
    })


@api_bp.route("/history", methods=["GET"])
def history():
    limit = request.args.get("limit", 50, type=int)
    rows = get_recent_predictions(limit)
    return jsonify(rows)


@api_bp.route("/stats", methods=["GET"])
def stats():
    return jsonify(get_stats())


@api_bp.route("/evaluation", methods=["GET"])
def evaluation():
    import os
    from config import BASE_DIR
    report_path = os.path.join(BASE_DIR, "figures", "evaluation_report.json")
    if not os.path.exists(report_path):
        return jsonify({"error": "Evaluation report not found. Run generate_figures.py first."}), 404
    with open(report_path, "r") as f:
        data = json.load(f)
    return jsonify(data)


@api_bp.route("/health", methods=["GET"])
def health():
    global vectorizer, classifier
    return jsonify({
        "status": "ok",
        "model_loaded": vectorizer is not None and classifier is not None,
    })
