import pandas as pd
import numpy as np
import joblib
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)

from config import (
    COMBINED_DATASET_PATH, MODELS_DIR, TFIDF_PARAMS, RF_PARAMS,
    RF_GRID, TRAIN_SPLIT, VAL_SPLIT,
)
from preprocessing.text_cleaner import clean_text


def load_and_sample_data(path, max_samples=100000):
    df = pd.read_csv(path)
    df["email_text"] = df["email_text"].astype(str)
    df["label"] = df["label"].astype(int)
    print(f"Loaded {len(df)} rows")

    vc = df["label"].value_counts()
    print(f"Labels: 0={vc.get(0,0)}  1={vc.get(1,0)}  ratio={vc.get(0,0)/max(vc.get(1,0),1):.2f}")

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"Sampled down to {max_samples} rows")

    return df


def run_training():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_and_sample_data(COMBINED_DATASET_PATH)

    print("Cleaning text...")
    t0 = time.time()
    df["clean_text"] = df["email_text"].apply(clean_text)
    print(f"Cleaning done in {time.time()-t0:.2f}s")

    X = df["clean_text"].values
    y = df["label"].values

    test_val_size = 1.0 - TRAIN_SPLIT
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_val_size, random_state=42, stratify=y
    )
    val_size = VAL_SPLIT / test_val_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1.0 - val_size, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    print("\nFitting TF-IDF vectorizer...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Vectorization done in {time.time()-t0:.2f}s  shape={X_train_vec.shape}")

    print("\nTraining Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train_vec, y_train)
    train_time = time.time() - t0
    print(f"Training done in {train_time:.2f}s")

    print("\n--- Evaluation ---")
    for name, X_eval, y_eval in [("Train", X_train_vec, y_train), ("Val", X_val_vec, y_val), ("Test", X_test_vec, y_test)]:
        y_pred = rf.predict(X_eval)
        y_prob = rf.predict_proba(X_eval)[:, 1]
        acc = accuracy_score(y_eval, y_pred)
        prec = precision_score(y_eval, y_pred)
        rec = recall_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred)
        auc = roc_auc_score(y_eval, y_prob)
        cm = confusion_matrix(y_eval, y_pred)
        print(f"\n{name} set:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

    cv_scores = cross_val_score(rf, X_train_vec, y_train, cv=5, scoring="accuracy")
    print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    print("\n--- Classification Report (Test) ---")
    y_test_pred = rf.predict(X_test_vec)
    print(classification_report(y_test, y_test_pred, target_names=["Legitimate", "Phishing"]))

    feature_names = vectorizer.get_feature_names_out()
    importances = rf.feature_importances_
    top_n = 25
    top_idx = np.argsort(importances)[-top_n:][::-1]
    print(f"\nTop {top_n} most important features:")
    for i in top_idx:
        print(f"  {feature_names[i]:25s} {importances[i]:.6f}")

    print("\nSaving model...")
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "rf_classifier.pkl"))
    print("Saved to models/tfidf_vectorizer.pkl and models/rf_classifier.pkl")


if __name__ == "__main__":
    run_training()
