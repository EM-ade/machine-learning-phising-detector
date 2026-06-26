import os, sys, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from preprocessing.text_cleaner import clean_text

FIGS_DIR = os.path.join(config.BASE_DIR, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.figsize": (10, 6),
    "font.family": "sans-serif",
})


def plot_confusion_matrix(cm, labels, title, fname):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel="True Label", xlabel="Predicted Label")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_roc_curve(y_true, y_score, title, fname):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random (AUC = 0.5)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_pr_curve(y_true, y_score, title, fname):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="green", lw=2, label=f"AP = {ap:.4f}")
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1.5, label=f"Baseline = {baseline:.3f}")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_feature_importance(feature_names, importances, top_n, title, fname):
    top_idx = np.argsort(importances)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_imps = [importances[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
    bars = ax.barh(range(top_n), top_imps, color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title, fontweight="bold")
    for bar, val in zip(bars, top_imps):
        ax.text(val + 0.0002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_class_distribution(labels, title, fname):
    unique, counts = np.unique(labels, return_counts=True)
    labels_str = ["Legitimate (0)", "Phishing (1)"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    colors_bar = ["#28a745", "#dc3545"]
    bars = ax1.bar(labels_str, counts, color=colors_bar, width=0.5, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Number of Emails")
    ax1.set_title("Class Distribution (Bar)", fontweight="bold")
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                 str(count), ha="center", va="bottom", fontweight="bold")

    wedges, texts, autotexts = ax2.pie(
        counts, labels=labels_str, autopct="%1.1f%%",
        colors=colors_bar, startangle=90, explode=(0.03, 0.03),
        textprops={"fontsize": 10}
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax2.set_title("Class Distribution (Pie)", fontweight="bold")

    fig.suptitle(title, fontweight="bold", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_text_length_distribution(df, title, fname):
    df_plot = df.copy()
    df_plot["length"] = df_plot["email_text"].str.len()
    df_plot["label_name"] = df_plot["label"].map({0: "Legitimate", 1: "Phishing"})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for label, color, name in [(0, "#28a745", "Legitimate"), (1, "#dc3545", "Phishing")]:
        subset = df_plot[df_plot["label"] == label]["length"]
        ax1.hist(subset, bins=80, alpha=0.6, color=color, label=name, density=True)
    ax1.set_xlabel("Text Length (characters)")
    ax1.set_ylabel("Density")
    ax1.set_title("Text Length Distribution", fontweight="bold")
    ax1.legend()
    ax1.set_xlim(0, min(df_plot["length"].quantile(0.98), 5000))

    df_plot.boxplot(column="length", by="label_name", ax=ax2, patch_artist=True,
                     boxprops=dict(facecolor="#4361ee", alpha=0.6))
    ax2.set_title("Text Length Box Plot", fontweight="bold")
    ax2.set_xlabel("")
    ax2.set_ylabel("Text Length (characters)")
    ax2.set_ylim(0, min(df_plot["length"].quantile(0.98), 5000))

    fig.suptitle(title, fontweight="bold", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_wordcloud(texts_phishing, texts_legitimate, title_prefix, fname_prefix):
    from wordcloud import WordCloud

    for label, texts, title_suffix, fname_suffix, color in [
        ("Phishing", texts_phishing, "Phishing Emails", "phishing", "Reds"),
        ("Legitimate", texts_legitimate, "Legitimate Emails", "legitimate", "Greens"),
    ]:
        combined = " ".join(texts.sample(min(5000, len(texts)), random_state=42))
        wc = WordCloud(
            width=1200, height=800,
            background_color="white",
            max_words=150,
            colormap=color,
            collocations=False,
            random_state=42,
        ).generate(combined)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{title_prefix}: {title_suffix}", fontweight="bold", fontsize=14, pad=15)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS_DIR, f"{fname_prefix}_{fname_suffix}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname_prefix}_{fname_suffix}.png")


def plot_cv_scores(scores, title, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(scores) + 1)
    bars = ax.bar(x, scores, color="#4361ee", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(y=np.mean(scores), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(scores):.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_ylim(min(scores) - 0.01, 1.0)
    ax.set_title(title, fontweight="bold")
    ax.legend()
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{score:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_metrics_comparison(metrics_dict, title, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    sets = list(metrics_dict.keys())
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    x = np.arange(len(metric_names))
    width = 0.25

    colors = {"Train": "#4361ee", "Val": "#f72585", "Test": "#16a34a"}
    for i, (s, vals) in enumerate(metrics_dict.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=s, color=colors.get(s, "#888"), alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score")
    ax.set_ylim(0.85, 1.0)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def generate_performance_plots():
    print("\n=== Generating Performance Plots ===")
    print("Loading model and test data...")
    vectorizer = joblib.load(os.path.join(config.MODELS_DIR, "tfidf_vectorizer.pkl"))
    classifier = joblib.load(os.path.join(config.MODELS_DIR, "rf_classifier.pkl"))

    df = pd.read_csv(config.COMBINED_DATASET_PATH)
    df["email_text"] = df["email_text"].astype(str)
    df = df.sample(n=30000, random_state=42)
    print(f"Loaded {len(df)} rows for evaluation")

    df["clean_text"] = df["email_text"].apply(clean_text)
    X = df["clean_text"].values
    y = df["label"].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    y_train_pred = classifier.predict(X_train_vec)
    y_test_pred = classifier.predict(X_test_vec)
    y_test_prob = classifier.predict_proba(X_test_vec)[:, 1]
    y_train_prob = classifier.predict_proba(X_train_vec)[:, 1]

    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, confusion_matrix, roc_auc_score)
    from sklearn.model_selection import cross_val_score

    cm_test = confusion_matrix(y_test, y_test_pred)
    cm_train = confusion_matrix(y_train, y_train_pred)

    plot_confusion_matrix(cm_test, ["Legitimate", "Phishing"],
                          "Confusion Matrix — Test Set", "confusion_matrix_test.png")
    plot_confusion_matrix(cm_train, ["Legitimate", "Phishing"],
                          "Confusion Matrix — Training Set", "confusion_matrix_train.png")

    plot_roc_curve(y_test, y_test_prob,
                   "ROC Curve — Test Set", "roc_curve_test.png")
    plot_roc_curve(y_train, y_train_prob,
                   "ROC Curve — Training Set", "roc_curve_train.png")

    plot_pr_curve(y_test, y_test_prob,
                  "Precision-Recall Curve — Test Set", "pr_curve_test.png")

    feature_names = vectorizer.get_feature_names_out()
    plot_feature_importance(feature_names, classifier.feature_importances_, 20,
                            "Top 20 Most Important Features", "feature_importance.png")

    metrics = {
        "Train": [
            accuracy_score(y_train, y_train_pred),
            precision_score(y_train, y_train_pred),
            recall_score(y_train, y_train_pred),
            f1_score(y_train, y_train_pred),
            roc_auc_score(y_train, y_train_prob),
        ],
        "Test": [
            accuracy_score(y_test, y_test_pred),
            precision_score(y_test, y_test_pred),
            recall_score(y_test, y_test_pred),
            f1_score(y_test, y_test_pred),
            roc_auc_score(y_test, y_test_prob),
        ],
    }
    plot_metrics_comparison(metrics,
                            "Model Performance: Train vs Test", "metrics_comparison.png")

    cv_scores = cross_val_score(classifier, X_train_vec, y_train, cv=5, scoring="accuracy")
    plot_cv_scores(cv_scores, "5-Fold Cross-Validation Accuracy", "cross_validation.png")


def generate_data_plots():
    print("\n=== Generating Data Analysis Plots ===")
    df = pd.read_csv(config.COMBINED_DATASET_PATH)
    df["email_text"] = df["email_text"].astype(str)
    df_sample = df.sample(n=50000, random_state=42) if len(df) > 50000 else df

    plot_class_distribution(df_sample["label"].values,
                            "Dataset Class Distribution", "class_distribution.png")

    plot_text_length_distribution(df_sample,
                                  "Email Text Length Analysis", "text_length_distribution.png")

    df_phish = df_sample[df_sample["label"] == 1]["email_text"]
    df_legit = df_sample[df_sample["label"] == 0]["email_text"]
    plot_wordcloud(df_phish, df_legit, "Word Cloud", "wordcloud")


def generate_architecture_diagram():
    print("\n=== Generating Architecture Diagrams ===")
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("System Architecture Diagram", fontweight="bold", fontsize=16, pad=20)

    layers = [
        {
            "y": 0.0, "h": 1.8, "color": "#e8f5e9", "edge": "#2e7d32",
            "title": "Data Layer",
            "items": ["SQLite Database", "Prediction Logs", "Model Artifacts (.pkl)"],
        },
        {
            "y": 2.0, "h": 2.0, "color": "#fff3e0", "edge": "#e65100",
            "title": "Processing Layer (ML Pipeline)",
            "items": ["NLP Preprocessor", "TF-IDF Vectorizer", "Random Forest Classifier"],
        },
        {
            "y": 4.3, "h": 2.0, "color": "#e3f2fd", "edge": "#1565c0",
            "title": "Application Layer (Flask Backend)",
            "items": ["REST API (/predict, /health)", "Route Handlers", "Model Inference Engine"],
        },
        {
            "y": 6.6, "h": 1.8, "color": "#fce4ec", "edge": "#c62828",
            "title": "Presentation Layer (Frontend)",
            "items": ["HTML5 + CSS3 UI", "JavaScript (Fetch API)", "Result Visualization"],
        },
    ]

    for i, layer in enumerate(reversed(layers)):
        y = layer["y"]
        h = layer["h"]
        rect = FancyBboxPatch((1, y), 12, h, boxstyle="round,pad=0.15",
                               facecolor=layer["color"], edgecolor=layer["edge"],
                               linewidth=2.5, zorder=2)
        ax.add_patch(rect)
        ax.text(7, y + h - 0.35, layer["title"], ha="center", va="center",
                fontweight="bold", fontsize=11, color=layer["edge"])

        items_text = "\n".join([f"  • {item}" for item in layer["items"]])
        ax.text(7, y + 0.4, items_text, ha="center", va="center",
                fontsize=9, color="#333", linespacing=1.5)

        if i < len(layers) - 1:
            ax.annotate("", xy=(7, layers[i+1]["y"] + layers[i+1]["h"]),
                        xytext=(7, y + h),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=2))

    ax.text(7, 9.2, "User (Web Browser)", ha="center", fontsize=12, fontweight="bold", color="#333")
    ax.annotate("", xy=(7, 8.4), xytext=(7, 9.0),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2.5))
    ax.text(7, -0.4, "HTTP Request / Response Flow", ha="center", fontsize=9, color="#666")

    fig.savefig(os.path.join(FIGS_DIR, "architecture_diagram.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved architecture_diagram.png")


def generate_workflow_diagram():
    print("  Generating workflow diagrams...")
    steps = [
        ("User Action", "Pastes email text\nclicks Analyze", "#e3f2fd", "#1565c0"),
        ("Frontend", "POST /predict\nJSON payload", "#fce4ec", "#c62828"),
        ("Flask Backend", "Validates input\nroutes request", "#fff3e0", "#e65100"),
        ("NLP Preprocessor", "Lowercase → Tokenize\n→ Stopwords → Lemmatize", "#f3e5f5", "#6a1b9a"),
        ("TF-IDF Vectorizer", "Transform text to\nnumerical features", "#e8f5e9", "#2e7d32"),
        ("Random Forest", "200 trees vote\n→ Prediction + Confidence", "#e8f5e9", "#2e7d32"),
        ("Response", "JSON result\n→ Frontend renders", "#fff3e0", "#e65100"),
        ("User View", "Sees result card\nwith indicators", "#e3f2fd", "#1565c0"),
    ]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, len(steps) + 0.5)
    ax.axis("off")
    ax.set_title("System Workflow / Sequence Diagram", fontweight="bold", fontsize=16, pad=15)

    for i, (title, desc, color, edge) in enumerate(steps):
        y = len(steps) - 1 - i
        rect = FancyBboxPatch((2, y - 0.4), 4, 0.8, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor=edge, linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(4, y, title, ha="center", va="center", fontweight="bold", fontsize=10, color=edge)

        rect2 = FancyBboxPatch((6.5, y - 0.4), 5.5, 0.8, boxstyle="round,pad=0.1",
                                facecolor="#fafafa", edgecolor="#ccc", linewidth=1.5, zorder=2)
        ax.add_patch(rect2)
        ax.text(9.25, y, desc, ha="center", va="center", fontsize=9, color="#333")

        if i < len(steps) - 1:
            next_y = len(steps) - 2 - i
            ax.annotate("", xy=(9.25, next_y + 0.4), xytext=(9.25, y - 0.4),
                        arrowprops=dict(arrowstyle="->", color="#888", lw=1.5, connectionstyle="arc3,rad=0"))

    fig.savefig(os.path.join(FIGS_DIR, "workflow_diagram.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved workflow_diagram.png")


def generate_class_diagram():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Class Diagram — System Components", fontweight="bold", fontsize=16, pad=15)

    classes = [
        {"x": 0.5, "y": 7, "w": 4, "h": 2.5, "color": "#e3f2fd", "edge": "#1565c0",
         "name": "WebInterface", "fields": ["- emailText: str", "- resultCard: DOM", "- historyTable: DOM"],
         "methods": ["+ handleSubmit(): void", "+ renderResult(data): void", "+ fetchHistory(): void"]},
        {"x": 5, "y": 7, "w": 4, "h": 2.5, "color": "#fff3e0", "edge": "#e65100",
         "name": "FlaskAPI", "fields": ["- app: Flask", "- vectorizer: TfidfVectorizer", "- classifier: RandomForest"],
         "methods": ["+ predict(email_text): JSON", "+ health(): JSON", "+ history(): JSON[]"]},
        {"x": 9.5, "y": 7, "w": 4, "h": 2.5, "color": "#e8f5e9", "edge": "#2e7d32",
         "name": "NLPPipeline", "fields": ["- stopwords: Set", "- lemmatizer: WordNetLemmatizer"],
         "methods": ["+ clean_text(raw): str", "+ tokenize(text): List[str]", "+ lemmatize(tokens): List[str]]"]},
        {"x": 0.5, "y": 3.5, "w": 4, "h": 2.5, "color": "#fce4ec", "edge": "#c62828",
         "name": "Database", "fields": ["- conn: SQLite3", "- path: str"],
         "methods": ["+ log_prediction(data): void", "+ get_history(limit): List", "+ get_stats(): Dict"]},
        {"x": 5, "y": 3.5, "w": 4, "h": 2.5, "color": "#f3e5f5", "edge": "#6a1b9a",
         "name": "ModelTrainer", "fields": ["- vectorizer: TfidfVectorizer", "- classifier: RandomForestClassifier"],
         "methods": ["+ load_data(): DataFrame", "+ train(X, y): void", "+ evaluate(X, y): Dict", "+ save(): void"]},
    ]

    for cls in classes:
        x, y, w, h = cls["x"], cls["y"], cls["w"], cls["h"]

        header_rect = FancyBboxPatch((x, y + h - 0.7), w, 0.7, boxstyle="round,pad=0.05",
                                      facecolor=cls["edge"], edgecolor=cls["edge"], linewidth=2)
        ax.add_patch(header_rect)
        ax.text(x + w/2, y + h - 0.35, f"<<class>>\n{cls['name']}", ha="center", va="center",
                fontweight="bold", fontsize=10, color="white")

        body_rect = FancyBboxPatch((x, y), w, h - 0.7, boxstyle="round,pad=0.05",
                                    facecolor=cls["color"], edgecolor=cls["edge"], linewidth=2)
        ax.add_patch(body_rect)

        fields_text = "\n".join(cls["fields"]) if cls["fields"] else ""
        ax.text(x + w/2, y + h - 1.3, fields_text, ha="center", va="top", fontsize=8,
                color="#555", family="monospace")

        divider_y = y + h - 1.5 if cls["fields"] else y + h - 0.7
        ax.axhline(y=divider_y, xmin=(x+0.2)/14, xmax=(x+w-0.2)/14, color=cls["edge"], linewidth=0.8, alpha=0.5)

        methods_text = "\n".join(cls["methods"]) if cls["methods"] else ""
        ax.text(x + w/2, divider_y - 0.1, methods_text, ha="center", va="top", fontsize=8,
                color="#333", family="monospace")

    relations = [
        (0.5+4, 8, 5, 8, "uses", "#666"),
        (5+4, 8, 9.5, 8, "uses", "#666"),
        (2.5, 7, 2.5, 6, "logs to", "#888"),
        (7, 7, 7, 6, "trains", "#888"),
    ]
    for x1, y1, x2, y2, label, color in relations:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5, connectionstyle="arc3,rad=0.15"))
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.1, my, label, fontsize=7, color=color, style="italic",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))

    fig.savefig(os.path.join(FIGS_DIR, "class_diagram.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved class_diagram.png")


def generate_data_flow_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Data Flow Diagram (DFD Level 0)", fontweight="bold", fontsize=16, pad=15)

    entities = [
        {"x": 0.5, "y": 3, "w": 3, "h": 1.5, "text": "User\n(End User)", "color": "#e3f2fd", "edge": "#1565c0"},
        {"x": 10.5, "y": 3, "w": 3, "h": 1.5, "text": "Dataset Repositories\n(PhishStorm, Enron, Kaggle)", "color": "#f3e5f5", "edge": "#6a1b9a"},
    ]
    for e in entities:
        rect = FancyBboxPatch((e["x"], e["y"]), e["w"], e["h"], boxstyle="round,pad=0.1",
                               facecolor=e["color"], edgecolor=e["edge"], linewidth=2.5)
        ax.add_patch(rect)
        ax.text(e["x"] + e["w"]/2, e["y"] + e["h"]/2, e["text"], ha="center", va="center",
                fontweight="bold", fontsize=10, color=e["edge"])

    processes = [
        {"x": 2, "y": 5.5, "w": 3.5, "h": 1.2, "text": "1. Data Loader\n(Merge & Clean)", "color": "#fff3e0", "edge": "#e65100"},
        {"x": 5.5, "y": 5.5, "w": 3.5, "h": 1.2, "text": "2. Preprocessor\n(TF-IDF Vectorize)", "color": "#e8f5e9", "edge": "#2e7d32"},
        {"x": 9, "y": 5.5, "w": 3.5, "h": 1.2, "text": "3. Classifier\n(Random Forest)", "color": "#fce4ec", "edge": "#c62828"},
        {"x": 3.5, "y": 1, "w": 3, "h": 1.2, "text": "4. Flask API\n(Predict)", "color": "#fff3e0", "edge": "#e65100"},
        {"x": 7.5, "y": 1, "w": 3, "h": 1.2, "text": "5. Database\n(SQLite Logs)", "color": "#e8f5e9", "edge": "#2e7d32"},
    ]
    for p in processes:
        rect = FancyBboxPatch((p["x"], p["y"]), p["w"], p["h"], boxstyle="round,pad=0.1",
                               facecolor=p["color"], edgecolor=p["edge"], linewidth=2)
        ax.add_patch(rect)
        ax.text(p["x"] + p["w"]/2, p["y"] + p["h"]/2, p["text"], ha="center", va="center",
                fontweight="bold", fontsize=9, color=p["edge"])

    arrows = [
        (3.75, 3, 3.75, 5.5, "Email text"),
        (5.5, 6.1, 7.25, 6.1, "Clean text"),
        (9, 6.1, 10.75, 6.1, "Feature vectors"),
        (7.5, 5.5, 7.5, 2.2, "Model (serialized)"),
        (5, 2.2, 3.5, 2.2, "Prediction"),
        (12, 3, 12, 5.5, "Training data"),
        (3.5, 1.6, 3.5, 3, "User request"),
        (7.5, 1, 7.5, 0.3, "Logs"),
    ]
    for x1, y1, x2, y2, label in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#888", lw=1.5, connectionstyle="arc3,rad=0.1"))
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, label, fontsize=7.5, color="#555", style="italic",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

    fig.savefig(os.path.join(FIGS_DIR, "data_flow_diagram.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved data_flow_diagram.png")


def generate_evaluation_report():
    print("  Generating evaluation report...")
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    import json

    vectorizer = joblib.load(os.path.join(config.MODELS_DIR, "tfidf_vectorizer.pkl"))
    classifier = joblib.load(os.path.join(config.MODELS_DIR, "rf_classifier.pkl"))

    df = pd.read_csv(config.COMBINED_DATASET_PATH)
    df = df.sample(n=30000, random_state=42)
    df["clean_text"] = df["email_text"].astype(str).apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"].values, df["label"].values, test_size=0.3, random_state=42, stratify=df["label"].values)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_vec)
    y_prob = classifier.predict_proba(X_test_vec)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"], output_dict=True)

    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y_test, y_prob)

    report_data = {
        "model_version": config.MODEL_VERSION,
        "test_samples": len(y_test),
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "true_neg": int(cm[0, 0]),
            "false_pos": int(cm[0, 1]),
            "false_neg": int(cm[1, 0]),
            "true_pos": int(cm[1, 1]),
        },
        "classification_report": {
            "legitimate": {
                "precision": report["Legitimate"]["precision"],
                "recall": report["Legitimate"]["recall"],
                "f1": report["Legitimate"]["f1-score"],
                "support": int(report["Legitimate"]["support"]),
            },
            "phishing": {
                "precision": report["Phishing"]["precision"],
                "recall": report["Phishing"]["recall"],
                "f1": report["Phishing"]["f1-score"],
                "support": int(report["Phishing"]["support"]),
            },
            "accuracy": report["accuracy"],
            "macro_avg": {
                "precision": report["macro avg"]["precision"],
                "recall": report["macro avg"]["recall"],
                "f1": report["macro avg"]["f1-score"],
            },
        },
    }

    with open(os.path.join(FIGS_DIR, "evaluation_report.json"), "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Saved evaluation_report.json")

    with open(os.path.join(FIGS_DIR, "evaluation_report.txt"), "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model Version: {config.MODEL_VERSION}\n")
        f.write(f"Test Samples:  {len(y_test)}\n\n")
        f.write("-" * 60 + "\n")
        f.write("  Confusion Matrix\n")
        f.write("-" * 60 + "\n")
        f.write(f"               Predicted\n")
        f.write(f"               Legit  Phish\n")
        f.write(f"Actual Legit   {cm[0,0]:>5}  {cm[0,1]:>5}\n")
        f.write(f"       Phish   {cm[1,0]:>5}  {cm[1,1]:>5}\n\n")
        f.write("-" * 60 + "\n")
        f.write("  Classification Report\n")
        f.write("-" * 60 + "\n")
        f.write(f"                Precision  Recall    F1-Score  Support\n")
        for cls_name in ["Legitimate", "Phishing"]:
            r = report[cls_name]
            f.write(f"  {cls_name:<12} {r['precision']:.4f}    {r['recall']:.4f}    {r['f1-score']:.4f}    {int(r['support']):>6}\n")
        f.write(f"\n  Accuracy:                 {report['accuracy']:.4f}\n")
        f.write(f"  Macro Avg Precision:      {report['macro avg']['precision']:.4f}\n")
        f.write(f"  Macro Avg Recall:         {report['macro avg']['recall']:.4f}\n")
        f.write(f"  Macro Avg F1:             {report['macro avg']['f1-score']:.4f}\n\n")
        f.write("=" * 60 + "\n")
    print(f"  Saved evaluation_report.txt")


def main():
    from sklearn.model_selection import train_test_split

    print("=" * 50)
    print("PhishGuard — Figure Generation Script")
    print("=" * 50)

    generate_data_plots()
    generate_performance_plots()
    generate_architecture_diagram()
    generate_workflow_diagram()
    generate_class_diagram()
    generate_data_flow_diagram()
    generate_evaluation_report()

    print("\n" + "=" * 50)
    print(f"All figures saved to: {FIGS_DIR}")
    print("=" * 50)

    files = os.listdir(FIGS_DIR)
    for f in sorted(files):
        fsize = os.path.getsize(os.path.join(FIGS_DIR, f))
        print(f"  {f:45s} {fsize/1024:>8.1f} KB")


if __name__ == "__main__":
    main()
