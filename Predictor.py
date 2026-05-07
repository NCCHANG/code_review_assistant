import joblib
import os
import sys

import numpy as np
from scipy.sparse import hstack, csr_matrix

# ── Path setup ───────────────────────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
rf_dir = os.path.join(script_dir, "training", "rf")

if rf_dir not in sys.path:
    sys.path.insert(0, rf_dir)

from training.rf.rf_model_utils import extract_ast_features, AST_FEATURE_NAMES

# Maps numeric class labels to human-readable bug type names
BUG_TYPE_NAMES = {
    0: "Clean",
    1: "Wrong Binary Operator",
    2: "Variable Misuse",
    3: "Swapped Operand",
}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    """Load the LightGBM classifier and TF-IDF vectorizer from training/rf/models/."""
    models_dir = os.path.join(rf_dir, "models")
    model_path = os.path.join(models_dir, "rf_classifier.joblib")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print(
            "Path Error: Model or vectorizer not found. "
            f"Expected files in '{models_dir}'."
        )
        return None, None

    print("Loading model...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model loaded.")
    return model, vectorizer


# ── Predictor class ───────────────────────────────────────────────────────────

class Predictor:
    def __init__(self):
        self.model, self.vectorizer = load_model()
        if self.model is None or self.vectorizer is None:
            print("Error: Model or vectorizer could not be loaded. Predictor will not function.")

    def predict(self, code_snippet: str) -> tuple[bool, float, str]:
        """Predict the bug type of *code_snippet* using multi-class classification.

        Builds a combined feature vector by horizontally stacking the TF-IDF
        sparse matrix with a 35-dimension AST feature vector, matching the
        feature space used during training.

        Returns
        -------
        (is_buggy, confidence, bug_type)
            is_buggy   – True if the predicted class is not Clean
            confidence – probability of the predicted class
            bug_type   – one of "Clean", "Wrong Binary Operator",
                         "Variable Misuse", "Swapped Operand"
        """
        # TF-IDF features
        tfidf_features = self.vectorizer.transform([code_snippet])

        # AST structural features (35 dimensions, see AST_FEATURE_NAMES)
        ast_feat_dict = extract_ast_features(code_snippet)
        ast_array = np.array(
            [[ast_feat_dict[name] for name in AST_FEATURE_NAMES]],
            dtype=np.float32,
        )
        ast_sparse = csr_matrix(ast_array)

        # Combine into the full feature vector expected by the model
        combined = hstack([tfidf_features, ast_sparse])

        # Pick the class with the highest predicted probability
        probs = self.model.predict_proba(combined)[0]
        best_idx = int(np.argmax(probs))
        predicted_class = int(self.model.classes_[best_idx])
        confidence = float(probs[best_idx])

        is_buggy = predicted_class != 0
        bug_type = BUG_TYPE_NAMES.get(predicted_class, f"Class {predicted_class}")

        return is_buggy, confidence, bug_type


if __name__ == "__main__":
    predictor = Predictor()
    if predictor.model and predictor.vectorizer:
        sample = "def add(a, b):\n    return a - b"
        is_buggy, confidence, bug_type = predictor.predict(sample)
        status = f"BUGGY ({bug_type})" if is_buggy else "CLEAN"
        print(f"Result: {status} (Confidence: {confidence:.2%})")
