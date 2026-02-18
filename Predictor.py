import joblib
import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure we can find model_utils in the training directory
# Ensure we can find model files in the training directory
training_dir = os.path.join(script_dir, "training")

# Ensure we can find rf_model_utils in the training/rf directory
rf_dir = os.path.join(training_dir, "rf")
if rf_dir not in sys.path:
    sys.path.append(rf_dir)

from rf_model_utils import tokenizer

def load_model():
    model_path = os.path.join(training_dir, "rf_model.joblib")
    vectorizer_path = os.path.join(training_dir, "vectorizer.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Model files not found. Please train the model first.")
        return None, None
        
    print("Loading model...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model loaded.")
    return model, vectorizer

def predict_bug(code_snippet, model, vectorizer, threshold=0.30):
    # Pass raw code snippet (vectorizer handles tokenization)
    vectorized_code = vectorizer.transform([code_snippet])
    
    # Predict Probability
    # Class 0 = Clean, Class 1 = Buggy
    probability = model.predict_proba(vectorized_code)[0][1] 
    
    # Custom Thresholding (Default 0.35 to prioritize Recall)
    prediction = 1 if probability >= threshold else 0
    
    return prediction, probability

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
                
        is_buggy, confidence = predict_bug(user_input, model, vectorizer)
        
        if is_buggy == 1:
            status = "BUGGY"
        else:
            status = "CLEAN"
        
        print(f"Result: {status} (Confidence: {confidence:.2%})")
