import pytest
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.text_cleaner import clean_text


@pytest.fixture(scope="session", autouse=True)
def setup_app():
    from app import init_app
    init_app()


class TestPreprocessing:
    def test_lowercase(self):
        result = clean_text("HELLO WORLD")
        assert result == "hello world"

    def test_url_removal(self):
        result = clean_text("Check http://evil.com/phish now")
        assert "http" not in result
        assert "evil" not in result

    def test_html_removal(self):
        result = clean_text("<b>Urgent</b> <script>alert(1)</script>")
        assert "script" not in result

    def test_special_chars_removal(self):
        result = clean_text("URGENT!!! Verify $$$ now")
        assert "!!!" not in result
        assert "$$$" not in result

    def test_stopword_removal(self):
        result = clean_text("this is a test email")
        assert "this" not in result.split()
        assert "is" not in result.split()

    def test_lemmatization(self):
        result = clean_text("studies running")
        assert "studies" not in result.split()
        assert "study" in result

    def test_empty_input(self):
        assert clean_text("") == ""

    def test_single_chars_removed(self):
        result = clean_text("a b c")
        assert result == ""

    def test_number_removal(self):
        result = clean_text("test123 456")
        assert "123" not in result
        assert "456" not in result

    def test_phishing_keywords_preserved(self):
        result = clean_text("urgent verify account password")
        assert "urgent" in result
        assert "verify" in result
        assert "account" in result

    def test_long_text_reduces_length(self):
        text = "the and is " * 500
        result = clean_text(text)
        assert len(result) < len(text)

    def test_mixed_case(self):
        result = clean_text("URGENT Click Here To Verify")
        assert result == "urgent click verify"


class TestModel:
    def test_model_files_exist(self):
        from config import MODELS_DIR
        vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        model_path = os.path.join(MODELS_DIR, "rf_classifier.pkl")
        assert os.path.exists(vec_path), f"Missing vectorizer at {vec_path}"
        assert os.path.exists(model_path), f"Missing model at {model_path}"

    def test_model_artifacts_loaded(self):
        import joblib
        from config import MODELS_DIR
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        classifier = joblib.load(os.path.join(MODELS_DIR, "rf_classifier.pkl"))
        assert hasattr(vectorizer, "transform")
        assert hasattr(classifier, "predict")
        assert hasattr(classifier, "predict_proba")

    def test_model_prediction_shape(self):
        import joblib
        from config import MODELS_DIR
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        classifier = joblib.load(os.path.join(MODELS_DIR, "rf_classifier.pkl"))
        sample = clean_text("urgent verify your account password click now")
        vec = vectorizer.transform([sample])
        pred = classifier.predict(vec)
        proba = classifier.predict_proba(vec)[0]
        assert pred.shape == (1,)
        assert proba.shape == (2,)
        assert abs(proba[0] + proba[1] - 1.0) < 0.001

    def test_phishing_detection(self):
        import joblib
        from config import MODELS_DIR
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        classifier = joblib.load(os.path.join(MODELS_DIR, "rf_classifier.pkl"))
        phishing_texts = [
            "Urgent your account has been compromised click here to verify your password immediately",
            "Dear customer your bank account has been suspended please login to confirm details",
            "Security alert unusual login detected verify your identity to avoid account restriction",
        ]
        for text in phishing_texts:
            cleaned = clean_text(text)
            vec = vectorizer.transform([cleaned])
            pred = classifier.predict(vec)[0]
            assert pred == 1, f"Expected phishing for: {text[:50]}"

    def test_legitimate_detection(self):
        import joblib
        from config import MODELS_DIR
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        classifier = joblib.load(os.path.join(MODELS_DIR, "rf_classifier.pkl"))
        legit_texts = [
            "thank team please find attached quarterly report review",
            "meeting scheduled friday discuss project update timeline",
            "thank email will forward document request soon",
            "conference rescheduled next month details follow",
        ]
        for text in legit_texts:
            cleaned = clean_text(text)
            vec = vectorizer.transform([cleaned])
            pred = classifier.predict(vec)[0]
            assert pred == 0, f"Expected legitimate for: {text[:50]}"

    def test_model_accuracy_above_90(self):
        import joblib
        import pandas as pd
        from sklearn.metrics import accuracy_score
        from config import COMBINED_DATASET_PATH, MODELS_DIR

        vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        classifier = joblib.load(os.path.join(MODELS_DIR, "rf_classifier.pkl"))

        df = pd.read_csv(COMBINED_DATASET_PATH)
        df = df.sample(n=5000, random_state=42)
        df["clean_text"] = df["email_text"].astype(str).apply(clean_text)

        X = vectorizer.transform(df["clean_text"].values)
        y_pred = classifier.predict(X)
        acc = accuracy_score(df["label"].values, y_pred)
        assert acc > 0.90, f"Accuracy {acc:.4f} < 0.90"

    def test_inference_speed(self):
        import joblib
        from config import MODELS_DIR
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        classifier = joblib.load(os.path.join(MODELS_DIR, "rf_classifier.pkl"))
        sample = clean_text(" ".join(["test"] * 200))
        times = []
        for _ in range(10):
            t0 = time.time()
            vec = vectorizer.transform([sample])
            classifier.predict(vec)
            times.append((time.time() - t0) * 1000)
        avg = np.mean(times)
        assert avg < 300, f"Avg inference too slow: {avg:.1f}ms"

    def test_feature_importance_nonzero(self):
        import joblib
        from config import MODELS_DIR
        classifier = joblib.load(os.path.join(MODELS_DIR, "rf_classifier.pkl"))
        importances = classifier.feature_importances_
        assert np.all(importances >= 0)
        assert np.sum(importances > 0) > 100


class TestAPI:
    @pytest.fixture
    def client(self):
        from app import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_predict_phishing(self, client):
        resp = client.post("/predict", json={
            "email_text": "Urgent your account has been compromised click here to verify immediately"
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["prediction"] == "Phishing"
        assert data["is_phishing"] is True
        assert data["confidence"] > 50

    def test_predict_legitimate(self, client):
        resp = client.post("/predict", json={
            "email_text": "thank team please find attached meeting notes today"
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["prediction"] == "Legitimate"
        assert data["is_phishing"] is False
        assert data["confidence"] > 50

    def test_predict_empty(self, client):
        resp = client.post("/predict", json={"email_text": ""})
        assert resp.status_code == 400

    def test_predict_missing_field(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 400

    def test_predict_response_schema(self, client):
        resp = client.post("/predict", json={"email_text": "test email content"})
        assert resp.status_code == 200
        data = resp.get_json()
        required = ["prediction", "confidence", "is_phishing", "indicators", "processing_time_ms"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_history_endpoint(self, client):
        client.post("/predict", json={"email_text": "test email content"})
        resp = client.get("/history")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)

    def test_history_after_prediction(self, client):
        client.post("/predict", json={"email_text": "urgent verify account"})
        resp = client.get("/history?limit=5")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) >= 1
        latest = data[0]
        assert "prediction" in latest
        assert "confidence" in latest
        assert "processing_time_ms" in latest

    def test_stats_endpoint(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_analyzed" in data

    def test_evaluation_endpoint(self, client):
        resp = client.get("/evaluation")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "model_version" in data
        assert "classification_report" in data
        assert "confusion_matrix" in data
        assert data["classification_report"]["accuracy"] > 0.90

    def test_index_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"PhishGuard" in resp.data

    def test_long_text_limit(self, client):
        long_text = "x" * 60000
        resp = client.post("/predict", json={"email_text": long_text})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data


class TestFigures:
    def test_figures_directory_exists(self):
        from config import BASE_DIR
        figs_dir = os.path.join(BASE_DIR, "figures")
        assert os.path.isdir(figs_dir)

    def test_required_figures_exist(self):
        from config import BASE_DIR
        figs_dir = os.path.join(BASE_DIR, "figures")
        required = [
            "class_distribution.png",
            "confusion_matrix_test.png",
            "roc_curve_test.png",
            "feature_importance.png",
            "architecture_diagram.png",
            "workflow_diagram.png",
            "class_diagram.png",
            "data_flow_diagram.png",
            "wordcloud_phishing.png",
            "wordcloud_legitimate.png",
            "cross_validation.png",
            "metrics_comparison.png",
            "pr_curve_test.png",
        ]
        for fname in required:
            path = os.path.join(figs_dir, fname)
            assert os.path.isfile(path), f"Missing required figure: {fname}"
            assert os.path.getsize(path) > 5000, f"Figure too small: {fname}"

    def test_evaluation_report_exists(self):
        from config import BASE_DIR
        report_path = os.path.join(BASE_DIR, "figures", "evaluation_report.json")
        assert os.path.isfile(report_path)
        import json
        with open(report_path) as f:
            report = json.load(f)
        assert "model_version" in report
        assert report["classification_report"]["accuracy"] > 0.90


class TestConfig:
    def test_config_has_required_params(self):
        import config
        assert config.MODEL_VERSION
        assert config.TFIDF_PARAMS["max_features"] == 5000
        assert config.RF_PARAMS["n_estimators"] == 200
        assert config.TRAIN_SPLIT > 0 and config.TEST_SPLIT > 0


class TestDatabase:
    def test_db_initializes(self):
        from database.db import init_db, get_db
        init_db()
        conn = get_db()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_db_log_and_retrieve(self):
        from database.db import init_db, log_prediction, get_recent_predictions, get_stats
        init_db()
        log_prediction("test email content", "Phishing", 95.5, True, 42.0, ["test"])
        history = get_recent_predictions(limit=1)
        assert len(history) >= 1
        latest = history[0]
        assert latest["prediction"] == "Phishing"
        assert latest["confidence"] == 95.5
        stats = get_stats()
        assert stats["total_analyzed"] >= 1
