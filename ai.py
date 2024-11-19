from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter

file_path = r"C:\Users\maria\Disease-Symptom-Extensive-Clean\Final_Augmented_dataset_Diseases_and_Symptoms.csv"
df = pd.read_csv(file_path)
base_dir = Path(__file__).parent

print(df.head())
print(df.columns)

sample_size = 500
df_sample = df.sample(n=sample_size, random_state=42)

if 'diseases' not in df_sample.columns:
    raise ValueError("Column 'diseases' does not exist in the dataset.")

X = df_sample.drop(columns=['diseases'])
y = df_sample['diseases']

print("Class distribution before filtering:", Counter(y))

min_class_size = 3
y_counts = Counter(y)
valid_classes = [cls for cls, count in y_counts.items() if count >= min_class_size]

if len(valid_classes) == 0:
    raise ValueError("No valid classes remain after filtering classes with fewer than 3 samples.")

df_filtered = df_sample[df_sample['diseases'].isin(valid_classes)]

X_filtered = df_filtered.drop(columns=['diseases'])
y_filtered = df_filtered['diseases']

if len(y_filtered) < 2:
    raise ValueError("Not enough samples left to apply SMOTE.")

smote = SMOTE(random_state=42, k_neighbors=2)
X_res, y_res = smote.fit_resample(X_filtered, y_filtered)

print(f"After SMOTE: {Counter(y_res)}")
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

cv_scores = cross_val_score(best_rf, X_res, y_res, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean()}")

best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"Balanced accuracy: {balanced_accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
model_path = base_dir / "wemlAi.joblib"

joblib.dump(best_rf, model_path);
