import joblib
import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

training_dir = os.path.join(script_dir, "training")
rf_dir = os.path.join(training_dir, "rf")

if rf_dir not in sys.path:
    sys.path.insert(0,rf_dir)

from training.rf.rf_model_utils import tokenizer

def load_model():
    model_path = os.path.join(rf_dir, "rf_model.joblib")
    vectorizer_path = os.path.join(rf_dir, "vectorizer.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Path Error: Model or vectorizer file not found. Please ensure 'rf_model.joblib' and 'vectorizer.joblib' are in the 'training' directory.")
        return None, None
        
    print("Loading model...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model loaded.")
    return model, vectorizer

class Predictor:
    model = None
    vectorizer = None
    def __init__(self):
        self.model, self.vectorizer = load_model()
        if self.model is None or self.vectorizer is None:
            print("Error: Model or vectorizer could not be loaded. Predictor will not function.")

    def predict(self, code_snippet: str, threshold=0.30):
        vectorized_code = self.vectorizer.transform([code_snippet])

        proba = self.model.predict_proba(vectorized_code)[0]
        classes = self.model.classes_

        max_idx = proba.argmax()
        predicted_class = classes[max_idx]
        confidence = proba[max_idx]

        if predicted_class == 'CLEAN' or confidence < threshold:
            return False, confidence, None
        return True, confidence, predicted_class

if __name__ == "__main__":
    model, vectorizer = load_model()
    
    if model and vectorizer:
        print("\n--- Interactive Predictor ---")
        print("Enter a code snippet to check (or 'exit' to quit):")
        user_input = "def add(a, b):\n    return a + b"
        
        # while True:
        #     user_input = input("\nCode >> ")
        #     if user_input.lower() == 'exit':
        #         break
                
        predictor = Predictor()
        is_buggy, confidence, bug_type = predictor.predict(user_input)

        if is_buggy:
            print(f"Result: BUGGY — {bug_type} (Confidence: {confidence:.2%})")
        else:
            print(f"Result: CLEAN (Confidence: {confidence:.2%})")
