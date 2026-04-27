"""Inference helper for the trained Random Forest bug classifier.

Exposes predict_function() for use as a pipeline gate: if the RF detects a bug
(label != 0) the function should be forwarded to CodeT5 for repair.

CLI usage:
    python predict.py --code "def add(a, b): return a - b"
"""

import argparse
import os
import sys
import warnings

import pandas as pd
from scipy.sparse import hstack

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from rf_model_utils import build_ast_feature_matrix

MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_classifier.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

LABEL_NAMES = {
    0: "Clean",
    1: "Wrong Binary Operator",
    2: "Variable Misuse",
    3: "Swapped Operand",
}

# Lazy-loaded singletons — loaded once on first predict_function() call.
_model = None
_vectorizer = None


def _load_artifacts():
    """Load model and vectorizer from disk (once) into module-level singletons."""
    global _model, _vectorizer
    if _model is not None:
        return
    for path in (RF_MODEL_PATH, VECTORIZER_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model artifact not found: {path}\n"
                "Run train_rf.py first to generate trained model files."
            )
    from joblib import load
    _model = load(RF_MODEL_PATH)
    _vectorizer = load(VECTORIZER_PATH)


def predict_function(code: str) -> dict:
    """Classify a Python function snippet and return a structured prediction.

    Args:
        code: Raw Python function source code as a string.

    Returns:
        {
            "label": int,           # 0=Clean, 1=WrongBinOp, 2=VarMisuse, 3=SwapOp
            "bug_type": str,        # human-readable class name
            "confidence": float,    # predicted class probability (0.0–1.0)
            "should_pass_to_codet5": bool  # True when a bug is detected
        }
    """
    _load_artifacts()
    tfidf_features = _vectorizer.transform([code])
    ast_features = build_ast_feature_matrix(pd.Series([code]))
    features = hstack([tfidf_features, ast_features])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        label = int(_model.predict(features)[0])
        proba = _model.predict_proba(features)[0]
    confidence = float(proba[label])
    return {
        "label": label,
        "bug_type": LABEL_NAMES[label],
        "confidence": round(confidence, 4),
        "should_pass_to_codet5": label != 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="RF bug classifier — gate for the CodeT5 repair pipeline"
    )
    parser.add_argument(
        "--code",
        required=True,
        help="Python function source code to classify (as a string)",
    )
    args = parser.parse_args()

    result = predict_function(args.code)
    print(f"Label            : {result['label']}")
    print(f"Bug type         : {result['bug_type']}")
    print(f"Confidence       : {result['confidence']:.4f}")
    print(f"Pass to CodeT5   : {result['should_pass_to_codet5']}")


if __name__ == "__main__":
    main()
