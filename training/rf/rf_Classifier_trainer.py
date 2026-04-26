import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

if script_dir not in sys.path:
    sys.path.append(script_dir)

from rf_model_utils import tokenizer

try:
    train_df = pd.read_csv(os.path.join(project_root, "processed_data", "rf", "rf_train.csv"))
    test_df = pd.read_csv(os.path.join(project_root, "processed_data", "rf", "rf_test.csv"))
except FileNotFoundError:
    print("Error: Dataset not found. Run splitData.py first to generate processed_data/rf/rf_train.csv and rf_test.csv.")
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
train_X = vectorizer.fit_transform(train_df['code'])
train_y = train_df['bug_type']
test_X = vectorizer.transform(test_df['code'])
test_y = test_df['bug_type']
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
