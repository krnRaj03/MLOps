import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score
import joblib
import os
from huggingface_hub import HfApi, hf_hub_download
import numpy as np

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)

print("📥 Downloading data from Hugging Face...\n")

# Download files
Xtrain_path = hf_hub_download(
    repo_id="Hugo014/Tourism-Package-Prediction",
    filename="Xtrain.csv",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)
Xtest_path = hf_hub_download(
    repo_id="Hugo014/Tourism-Package-Prediction",
    filename="Xtest.csv",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)
ytrain_path = hf_hub_download(
    repo_id="Hugo014/Tourism-Package-Prediction",
    filename="ytrain.csv",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)
ytest_path = hf_hub_download(
    repo_id="Hugo014/Tourism-Package-Prediction",
    filename="ytest.csv",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)

# Load data
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).values.ravel()
ytest = pd.read_csv(ytest_path).values.ravel()

print(f"✓ Data loaded: Train {Xtrain.shape}, Test {Xtest.shape}\n")

numeric_features = Xtrain.columns.tolist()

# Class weight
class_counts = pd.Series(ytrain).value_counts()
class_weight = class_counts[0] / class_counts[1]
print(f"✓ Class weight: {class_weight:.2f}\n")

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    remainder='passthrough'
)

# XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# EXPANDED hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [150, 200, 300],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.7, 0.8, 0.9],
    'xgbclassifier__colsample_bytree': [0.7, 0.8, 0.9],
    'xgbclassifier__min_child_weight': [1, 3, 5],
    'xgbclassifier__gamma': [0, 0.1, 0.2],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

print("🔍 Starting EXTENSIVE Grid Search...")
print(f"Total combinations: {3*3*3*3*3*3*3} = 2187")
print("This will take 30-60 minutes...\n")

# Use RandomizedSearchCV for faster tuning
grid_search = RandomizedSearchCV(
    model_pipeline,
    param_grid,
    n_iter=50,  # Test 50 random combinations instead of all 2187
    cv=5,
    scoring='recall',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_

print("\n" + "="*60)
print("✅ HYPERTUNING COMPLETE")
print("="*60)
print("\nBest Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  • {param}: {value}")
print(f"\nBest CV Recall: {grid_search.best_score_:.4f}")

# Predictions
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

# Comprehensive evaluation
train_acc = accuracy_score(ytrain, y_pred_train)
train_recall = recall_score(ytrain, y_pred_train)
train_f1 = f1_score(ytrain, y_pred_train)

test_acc = accuracy_score(ytest, y_pred_test)
test_recall = recall_score(ytest, y_pred_test)
test_f1 = f1_score(ytest, y_pred_test)

print("\n" + "="*60)
print("📊 MODEL PERFORMANCE COMPARISON")
print("="*60)
print("\nTRAINING SET:")
print(f"  Accuracy:  {train_acc:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")

print("\nTEST SET:")
print(f"  Accuracy:  {test_acc:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")

print("\nIMPROVEMENT vs V1 (Recall: 0.7879):")
improvement = ((test_recall - 0.7879) / 0.7879) * 100
print(f"  Recall Change: {improvement:+.2f}%")

print("\nDetailed Classification Report:")
print(classification_report(ytest, y_pred_test))

# Save model with version 2
model_filename = "best_tourism_model_v2.joblib"
joblib.dump(best_model, model_filename)
print(f"\n💾 Model saved: {model_filename}")

# Upload to HF
print(f"\n📤 Uploading to Hugging Face...")
api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo=model_filename,
    repo_id="Hugo014/Tourism-Model",
    repo_type="model",
    token=os.environ["HF_TOKEN"]
)

print(f"✅ Version 2 uploaded!")
print(f"\n🎉 Model: https://huggingface.co/Hugo014/Tourism-Model")

print("\n" + "="*60)
print("FINAL SUMMARY - VERSION 2")
print("="*60)
print(f"Test Recall: {test_recall:.4f} (Target metric)")
print(f"Test F1: {test_f1:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"CV Score: {grid_search.best_score_:.4f}")
print("="*60)
