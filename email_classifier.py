import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from catboost import CatBoostClassifier 
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
csv_path = os.path.join(BASE_DIR, "spam.csv")
data = pd.read_csv(csv_path, encoding="latin-1", on_bad_lines="skip")
# Assuming the CSV has columns 'v1' for labels and 'v2' for text
# Rename columns for convenience
if data.columns[0] == 'v1' and data.columns[1] == 'v2':
    data = data.rename(columns={'v1': 'label', 'v2': 'text'})
else:
    # Handle the case where the column names might be different or have extra spaces
    data.columns = ['label', 'text']
# Drop any rows with missing values
data = data.dropna(subset=['label', 'text'])
# Encode labels: 'ham' -> 0, 'spam' -> 1
data['label'] = data['label'].str.strip().map({'ham': 0, 'spam': 1})
# Extract features and labels
X = data['text']
y = data['label']
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Print the shapes of the training and testing sets
print("Shape of the training set:", X_train.shape, y_train.shape)
print("Shape of the testing set:", X_test.shape, y_test.shape)
# Vectorize the text data using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Create CatBoost model
model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1, loss_function='Logloss', verbose=100)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
