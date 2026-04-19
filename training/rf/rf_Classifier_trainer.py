import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import sys
import os
from rf_model_utils import tokenizer

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))


try:
    train_df = pd.read_csv("datasets/rf_train_dataset.csv")
    test_df = pd.read_csv("datasets/rf_test_dataset.csv")
except FileNotFoundError as e:
    print("Error: Error in finding training/testing dataset for Random Forest.")
    print(f"Error details: {e}")
    exit()

# pass the tokenizer to TfidfVectorizer and let it handle raw text.

vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 4),
    max_features=15000,
    min_df=3,
    # using custom tokenizer here
    tokenizer=tokenizer
)

print("vectorizing...")
# vectorize use custom tokenizer and transform the raw code snippets into tfidf features
train_X = vectorizer.fit_transform(train_df['code_snippet'])
train_y = train_df['label']
test_X = vectorizer.transform(test_df['code_snippet'])
test_y = test_df['label']
print("vectorizing done")

#TRAINING THE MODEL
print("training...")
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(train_X, train_y)
print("training done")

#EVALUATION
print("evaluating...")
y_pred = model.predict(test_X)
print("evaluating done")

print("Accuracy:", accuracy_score(test_y, y_pred))
print("Classification Report:")
print(classification_report(test_y, y_pred))

# Save the model and vectorizer
print("Saving model...")
dump(model, os.path.join(script_dir, "rf_model.joblib"))
dump(vectorizer, os.path.join(script_dir, "vectorizer.joblib"))
print(f"Model saved to {os.path.join(script_dir, 'rf_model.joblib')}")
