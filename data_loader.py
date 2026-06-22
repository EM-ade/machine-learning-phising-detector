import pandas as pd
import os
import re
from config import RAW_DATA_DIR, COMBINED_DATASET_PATH


def load_enron(fp):
    df = pd.read_csv(fp)
    df["email_text"] = df["body"].fillna("")
    df["label"] = df["label"].astype(int)
    return df[["email_text", "label"]]


def load_ceas(fp):
    df = pd.read_csv(fp)
    df["email_text"] = df[["subject", "body"]].fillna("").agg(" ".join, axis=1).str.strip()
    df["label"] = df["label"].astype(int)
    return df[["email_text", "label"]]


def load_nazario(fp):
    df = pd.read_csv(fp)
    df["email_text"] = df[["subject", "body"]].fillna("").agg(" ".join, axis=1).str.strip()
    df["label"] = 1
    return df[["email_text", "label"]]


def load_nigerian_fraud(fp):
    df = pd.read_csv(fp)
    df["email_text"] = df[["subject", "body"]].fillna("").agg(" ".join, axis=1).str.strip()
    df["label"] = 1
    return df[["email_text", "label"]]


def load_spamassasin(fp):
    df = pd.read_csv(fp)
    df["email_text"] = df[["subject", "body"]].fillna("").agg(" ".join, axis=1).str.strip()
    df["label"] = df["label"].astype(int)
    return df[["email_text", "label"]]


def load_ling(fp):
    df = pd.read_csv(fp)
    df["email_text"] = df["body"].fillna("")
    df["label"] = df["label"].astype(int)
    return df[["email_text", "label"]]


def load_phishing_email(fp):
    df = pd.read_csv(fp)
    df["email_text"] = df["Email Text"].fillna("")
    df["label"] = df["Email Type"].map({"Safe Email": 0, "Phishing Email": 1})
    return df[["email_text", "label"]]


def load_large_phishing(fp):
    df = pd.read_csv(fp)
    df["email_text"] = df["text_combined"].fillna("")
    df["label"] = df["label"].astype(int)
    return df[["email_text", "label"]]


def combine_datasets():
    loaders = [
        ("Enron.csv", load_enron),
        ("CEAS_08.csv", load_ceas),
        ("Ling.csv", load_ling),
        ("Nazario.csv", load_nazario),
        ("Nigerian_Fraud.csv", load_nigerian_fraud),
        ("SpamAssasin.csv", load_spamassasin),
        ("Phishing_Email.csv", load_phishing_email),
    ]

    frames = []
    for fname, loader in loaders:
        fp = os.path.join(RAW_DATA_DIR, fname)
        if not os.path.exists(fp):
            print(f"[SKIP] {fname} not found")
            continue
        try:
            df = loader(fp)
            frames.append(df)
            print(f"[OK]   {fname} -> {len(df)} rows")
        except Exception as e:
            print(f"[FAIL] {fname} -> {e}")

    if not frames:
        print("No data loaded. Exiting.")
        return

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {len(combined)} rows")

    combined = combined.dropna(subset=["email_text"])
    combined = combined[combined["email_text"].str.strip() != ""]
    combined = combined.drop_duplicates(subset=["email_text"])
    print(f"After dedup + drop empty: {len(combined)} rows")

    combined["email_text"] = combined["email_text"].astype(str)
    combined["label"] = combined["label"].astype(int)

    vc = combined["label"].value_counts()
    print(f"Label distribution: 0={vc.get(0,0)}  1={vc.get(1,0)}")

    combined.to_csv(COMBINED_DATASET_PATH, index=False)
    print(f"Saved to {COMBINED_DATASET_PATH}")
    return combined


if __name__ == "__main__":
    combine_datasets()
