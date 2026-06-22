import sqlite3
import hashlib
import json
import os
from datetime import datetime

from config import DATABASE_PATH


def get_db():
    os.makedirs(os.path.dirname(DATABASE_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (datetime('now','localtime')),
            input_hash TEXT,
            text_length INTEGER,
            prediction TEXT,
            confidence REAL,
            is_phishing INTEGER,
            processing_time_ms REAL,
            indicators TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
        ON predictions(timestamp DESC)
    """)
    conn.commit()
    conn.close()


def log_prediction(email_text, prediction, confidence, is_phishing, processing_time_ms, indicators):
    conn = get_db()
    input_hash = hashlib.sha256(email_text.encode("utf-8")).hexdigest()[:16]
    conn.execute(
        """INSERT INTO predictions
           (input_hash, text_length, prediction, confidence, is_phishing, processing_time_ms, indicators)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (input_hash, len(email_text), prediction, confidence, int(is_phishing), processing_time_ms, json.dumps(indicators)),
    )
    conn.commit()
    conn.close()


def get_recent_predictions(limit=50):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, timestamp, prediction, confidence, is_phishing, processing_time_ms, indicators, text_length "
        "FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
    phishing_count = conn.execute("SELECT COUNT(*) as c FROM predictions WHERE is_phishing=1").fetchone()["c"]
    avg_conf = conn.execute("SELECT AVG(confidence) as c FROM predictions").fetchone()["c"] or 0
    avg_time = conn.execute("SELECT AVG(processing_time_ms) as t FROM predictions").fetchone()["t"] or 0
    conn.close()
    return {
        "total_analyzed": total,
        "phishing_count": phishing_count,
        "avg_confidence": round(avg_conf, 2),
        "avg_processing_time_ms": round(avg_time, 2),
    }
