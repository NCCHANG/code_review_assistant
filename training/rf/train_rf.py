"""Train a LightGBM classifier on combined TF-IDF + AST features.

LightGBM (gradient boosting) outperforms Random Forest on sparse high-dimensional
data because it uses ALL features at every split via information-gain ranking,
whereas RF randomly sub-samples sqrt(n_features) ≈ 141 of 20032 — which means
the 32 AST features were almost never selected.

Outputs:
    models/rf_classifier.joblib     (LightGBM model, kept same name for pipeline compat)
    models/tfidf_vectorizer.joblib
"""

import os
import sys
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("LightGBM not installed. Run: pip install lightgbm")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from rf_model_utils import bug_aware_tokenizer, build_ast_feature_matrix

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "rf_classifier.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

LABEL_NAMES = ["Clean", "Wrong Binary Operator", "Variable Misuse", "Swapped Operand"]


def load_data():
    """Load train and test CSVs, exiting early if either is missing."""
    for path in (TRAIN_FILE, TEST_FILE):
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run split_dataset.py first.")
            sys.exit(1)
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    print(f"Train: {len(train):,} samples | Test: {len(test):,} samples")
    return train, test


def build_vectorizer():
    """Return a TF-IDF vectorizer configured for bug-aware code tokenization."""
    return TfidfVectorizer(
        tokenizer=bug_aware_tokenizer,
        max_features=20_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        analyzer="word",
    )


def train_model(X_train, y_train):
    """Fit and return a LightGBM classifier with class balancing."""
    clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=127,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def print_confusion_matrix(cm):
    col_width = 22
    header = " " * col_width + "".join(f"{n[:col_width]:>{col_width}}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        label = f"{LABEL_NAMES[i][:col_width]:>{col_width}}"
        values = "".join(f"{v:>{col_width}}" for v in row)
        print(label + values)


def evaluate(clf, X_test, y_test):
    """Print accuracy, per-class precision/recall/F1, and the confusion matrix."""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, digits=4))
    print("Confusion Matrix (rows = actual, cols = predicted):")
    print_confusion_matrix(confusion_matrix(y_test, y_pred))


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    train_df, test_df = load_data()

    # ── Feature extraction ───────────────────────────────────────────────────
    print("\nFitting TF-IDF vectorizer...")
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(train_df["function"])
    X_test_tfidf = vectorizer.transform(test_df["function"])
    print(f"TF-IDF matrix: train={X_train_tfidf.shape}, test={X_test_tfidf.shape}")

    print("Extracting AST features...")
    X_train_ast = build_ast_feature_matrix(train_df["function"])
    X_test_ast = build_ast_feature_matrix(test_df["function"])

    X_train = hstack([X_train_tfidf, X_train_ast]).tocsr()
    X_test = hstack([X_test_tfidf, X_test_ast]).tocsr()
    print(f"Combined matrix: train={X_train.shape}, test={X_test.shape}")

    # ── Training ─────────────────────────────────────────────────────────────
    print("\nTraining LightGBM (n_estimators=500, lr=0.05, num_leaves=127)...")
    clf = train_model(X_train, train_df["label"])
    print("Training complete.")

    # ── Evaluation ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    evaluate(clf, X_test, test_df["label"])

    # ── Save artefacts ───────────────────────────────────────────────────────
    dump(clf, MODEL_PATH)
    dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved model     : {MODEL_PATH}")
    print(f"Saved vectorizer: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
