# PhishGuard — Machine Learning-Based Phishing Email Classification System

A web-based application that uses **TF-IDF + Random Forest** to classify email text as **phishing** or **legitimate** in real time, with >98% accuracy.

**Research Project:** B.Sc. Cybersecurity, Chrisland University, Abeokuta (2026)
**Author:** Babayemi Ayoola (CYB/2022/004)

---

## Project Structure

```
phishing-detector/
├── app.py                           # Flask entry point (run this)
├── config.py                        # Paths, model parameters, thresholds
├── requirements.txt                 # Python dependencies
├── data_loader.py                   # Merges CSV datasets → unified CSV
├── train.py                         # Train TF-IDF + Random Forest model
├── generate_figures.py              # Generates all plots, diagrams, reports
├── render.yaml                      # Render deployment config
├── Procfile                         # Heroku/Railway deployment config
├── README.md                        # This file
│
├── data/
│   ├── raw/                         # Source CSV datasets (7 sources)
│   │   ├── CEAS_08.csv
│   │   ├── Enron.csv
│   │   ├── Ling.csv
│   │   ├── Nazario.csv
│   │   ├── Nigerian_Fraud.csv
│   │   ├── Phishing_Email.csv
│   │   └── SpamAssasin.csv
│   └── combined_dataset.csv         # Unified dataset (generated, not in Git)
│
├── models/                          # Trained artifacts (generated, not in Git)
│   ├── tfidf_vectorizer.pkl
│   └── rf_classifier.pkl
│
├── figures/                         # All project figures (generated)
│   ├── architecture_diagram.png     # System architecture (4-layer)
│   ├── workflow_diagram.png         # System workflow
│   ├── class_diagram.png            # UML class diagram
│   ├── data_flow_diagram.png        # DFD Level 0
│   ├── class_distribution.png       # Dataset balance (bar + pie)
│   ├── text_length_distribution.png # Histogram + box plot
│   ├── wordcloud_phishing.png       # Word cloud — phishing emails
│   ├── wordcloud_legitimate.png     # Word cloud — legitimate emails
│   ├── confusion_matrix_test.png    # Confusion matrix — test set
│   ├── roc_curve_test.png           # ROC curve — test set
│   ├── pr_curve_test.png            # Precision-Recall curve
│   ├── feature_importance.png       # Top 20 features
│   ├── cross_validation.png         # 5-fold CV scores
│   ├── metrics_comparison.png       # Train vs Test metrics
│   ├── evaluation_report.json       # Metrics in JSON
│   └── evaluation_report.txt        # Metrics in text
│
├── preprocessing/
│   ├── __init__.py
│   └── text_cleaner.py              # NLP pipeline: tokenize, lemmatize, etc.
│
├── routes/
│   ├── __init__.py
│   └── api.py                       # /predict, /health, /history, /stats, /evaluation
│
├── database/
│   ├── __init__.py
│   └── db.py                        # SQLite schema + query functions
│
├── static/
│   ├── css/style.css                # Responsive dark/light UI
│   └── js/app.js                    # Fetch API, result rendering
│
├── templates/
│   └── index.html                   # Single-page application
│
├── tests/
│   └── test_app.py                  # 38 tests across 6 test classes
│
├── phishing_detector.db             # SQLite database (auto-generated, not in Git)
└── .gitignore                       # Excludes large generated files
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10+
- pip (Python package manager)

### 2. Install Dependencies

```bash
cd phishing-detector
pip install -r requirements.txt
```

### 3. Combine Datasets (if not already done)

```bash
python data_loader.py
```

Merges all 8 CSV datasets into `data/combined_dataset.csv`.

### 4. Train the Model

```bash
python -u train.py
```

This runs:
- Load 182K+ emails
- Clean/preprocess all text
- TF-IDF vectorization (5,000 features)
- Random Forest training (200 trees)
- Evaluation on test set
- Save models to `models/`

**Training time:** ~4 minutes on a modern laptop.

### 5. Generate Figures (Plots & Diagrams)

```bash
python -u generate_figures.py
```

Generates 18+ figures in the `figures/` directory:
- Performance plots (confusion matrix, ROC, feature importance, etc.)
- Data analysis (class distribution, word clouds, text lengths)
- System diagrams (architecture, workflow, class, data flow)

### 6. Run the Web Application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Usage

1. **Paste email text** into the textarea
2. Click **Analyze** (or press **Ctrl+Enter**)
3. View result:
   - **🔴 PHISHING** — Red card with warning indicators
   - **🟢 LEGITIMATE** — Green card with confidence score
4. Check **Recent Analyses** table for prediction history

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| POST | `/predict` | Classify email text |
| GET | `/health` | Service health check |
| GET | `/history?limit=50` | Recent predictions |
| GET | `/stats` | Aggregate statistics |

### POST /predict

**Request:**
```json
{
  "email_text": "Urgent! Your account has been compromised. Click here to verify."
}
```

**Response:**
```json
{
  "prediction": "Phishing",
  "confidence": 97.34,
  "is_phishing": true,
  "indicators": [
    "Urgent language detected",
    "Credential request pattern",
    "Authority impersonation"
  ],
  "processing_time_ms": 42.15
}
```

---

## Testing

```bash
python -m pytest tests/test_app.py -v
```

37 tests across 6 categories:
| Test Class | Tests | What It Covers |
|------------|-------|----------------|
| TestPreprocessing | 12 | Text cleaning, lemmatization, URL/HTML removal |
| TestModel | 8 | Model files, prediction, accuracy >90%, speed |
| TestAPI | 11 | All endpoints, validation, response schema |
| TestFigures | 3 | All figures exist and are non-empty |
| TestConfig | 1 | Configuration parameters |
| TestDatabase | 2 | SQLite init, log/retrieve |

---

## Model Performance

| Metric | Value | Target |
|--------|-------|--------|
| **Accuracy** | **98.35%** | >90% ✓ |
| **Precision** | **99%** | >88% ✓ |
| **Recall** | **98%** | >88% ✓ |
| **F1 Score** | **98%** | >88% ✓ |
| **AUC** | **99.78%** | — |
| **Avg Inference** | **<50ms** | <200ms ✓ |

---

## Datasets Used (8 sources, 182K emails)

| Dataset | Source | Rows | Content |
|---------|--------|------|---------|
| Phishing Email | Kaggle (subhajournal) | 18,650 | Safe/Phishing labeled |
| Large Phishing | Kaggle | 82,486 | Labeled 0/1 |
| CEAS 2008 | CEAS Corpus | 39,154 | Phishing/legitimate |
| Enron | CMU Enron Corpus | 29,767 | Legitimate emails |
| SpamAssassin | Apache | 5,809 | Spam/ham |
| Nigerian Fraud | Open Source | 3,332 | Fraud emails |
| Ling | Linguistic Dataset | 2,859 | Labeled |
| Nazario | Nazario Corpus | 1,565 | Phishing samples |

---

## Deployment

### Quick Start (Local)

```bash
cd phishing-detector
pip install -r requirements.txt
python data_loader.py
python -u train.py
python app.py
```

Open **http://localhost:5000**

---

### Deploy to Render (Free)

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "PhishGuard v1.0"
git remote add origin https://github.com/your-username/phishing-detector.git
git push -u origin main
```

2. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

3. **Create Web Service**
   - Click "New" → "Web Service"
   - Connect your GitHub repo
   - Render auto-detects `render.yaml`

4. **Settings (auto-configured by render.yaml)**
   - **Build:** `pip install -r requirements.txt && python data_loader.py && python -u train.py`
   - **Start:** `gunicorn -w 2 -b 0.0.0.0:$PORT app:app`
   - **Health Check:** `/health`

5. **Deploy**
   - Click "Create Web Service"
   - First deploy: ~5 minutes (training)
   - Subsequent deploys: ~30 seconds (cached)

6. **Your app is live at:** `https://phishguard.onrender.com`

---

### Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

---

### Deploy with Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python data_loader.py && python -u train.py
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t phishguard .
docker run -p 5000:5000 phishguard
```

---

## Architecture

The system uses a **multi-tier architecture**:

```
┌────────────────────────────────────────────────┐
│  Presentation Layer (HTML/CSS/JS)              │
│  → Text input, result cards, history table     │
├────────────────────────────────────────────────┤
│  Application Layer (Flask Backend)             │
│  → REST API, routing, model inference          │
├────────────────────────────────────────────────┤
│  Processing Layer (ML Pipeline)                │
│  → NLP Preprocessor → TF-IDF → Random Forest  │
├────────────────────────────────────────────────┤
│  Data Layer (SQLite)                           │
│  → Prediction logs, performance metrics        │
└────────────────────────────────────────────────┘
```

See `figures/architecture_diagram.png` for full visualization.

---

## Key Files for Project Report

| Figure | File | Description |
|--------|------|-------------|
| System Architecture | `figures/architecture_diagram.png` | Multi-tier block diagram |
| System Workflow | `figures/workflow_diagram.png` | Step-by-step flow |
| Class Diagram | `figures/class_diagram.png` | UML component classes |
| Data Flow | `figures/data_flow_diagram.png` | DFD Level 0 |
| Confusion Matrix | `figures/confusion_matrix_test.png` | Test set performance |
| ROC Curve | `figures/roc_curve_test.png` | AUC = 99.78% |
| Feature Importance | `figures/feature_importance.png` | Top 20 words |
| Word Clouds | `figures/wordcloud_*.png` | Phishing vs Legitimate |
| Class Distribution | `figures/class_distribution.png` | Dataset balance |
| Evaluation Report | `figures/evaluation_report.txt` | Full metrics summary |
