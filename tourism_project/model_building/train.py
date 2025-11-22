# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

# Set token
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)

# Download files from HF using hf_hub_download (proper way)
print("Downloading files from Hugging Face...\n")

Xtrain_path = hf_hub_download(
    repo_id="Hugo014/Tourism-Package-Prediction",
    filename="Xtrain.csv",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)
print(f"✓ Downloaded Xtrain.csv")

Xtest_path = hf_hub_download(
    repo_id="Hugo014/Tourism-Package-Prediction",
    filename="Xtest.csv",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)
print(f"✓ Downloaded Xtest.csv")

ytrain_path = hf_hub_download(
    repo_id="Hugo014/Tourism-Package-Prediction",
    filename="ytrain.csv",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)
print(f"✓ Downloaded ytrain.csv")

ytest_path = hf_hub_download(
    repo_id="Hugo014/Tourism-Package-Prediction",
    filename="ytest.csv",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)
print(f"✓ Downloaded ytest.csv")

# Now load the downloaded files
print("\n📂 Loading data...")
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).values.ravel()
ytest = pd.read_csv(ytest_path).values.ravel()

print(f"✓ Training data: {Xtrain.shape}")
print(f"✓ Test data: {Xtest.shape}")

# All features are already encoded
numeric_features = Xtrain.columns.tolist()
print(f"✓ Features: {len(numeric_features)} columns")

# Class weight
class_counts = pd.Series(ytrain).value_counts()
class_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1
print(f"✓ Class weight: {class_weight:.2f}")

# Preprocessing pipeline
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

# Parameter grid
param_grid = {
    'xgbclassifier__n_estimators': [100, 150],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__subsample': [0.8],
    'xgbclassifier__colsample_bytree': [0.8],
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

print("\n🔍 Starting Grid Search (this will take 5-10 minutes)...\n")

# Grid search
grid_search = GridSearchCV(
    model_pipeline, 
    param_grid, 
    cv=5, 
    scoring='recall',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(Xtrain, ytrain)

# Best model
best_model = grid_search.best_estimator_
print("\n" + "="*50)
print("✅ TRAINING COMPLETE")
print("="*50)
print("\nBest Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  • {param}: {value}")
print(f"\nBest CV Recall: {grid_search.best_score_:.4f}")

# Predictions
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

# Evaluation
print("\n" + "="*50)
print("📊 TRAINING SET")
print("="*50)
train_acc = accuracy_score(ytrain, y_pred_train)
train_recall = recall_score(ytrain, y_pred_train)
print(f"Accuracy: {train_acc:.4f}")
print(f"Recall: {train_recall:.4f}")

print("\n" + "="*50)
print("📊 TEST SET")
print("="*50)
test_acc = accuracy_score(ytest, y_pred_test)
test_recall = recall_score(ytest, y_pred_test)
print(f"Accuracy: {test_acc:.4f}")
print(f"Recall: {test_recall:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(ytest, y_pred_test))

# Save model
model_filename = "best_tourism_model_v1.joblib"
joblib.dump(best_model, model_filename)
print(f"\n💾 Model saved: {model_filename}")

# Upload to HF
repo_id = "Hugo014/Tourism-Model"
repo_type = "model"

print(f"\n📤 Uploading to Hugging Face...")

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=os.environ["HF_TOKEN"])
    print("✓ Repository created")

api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo=model_filename,
    repo_id=repo_id,
    repo_type=repo_type,
    token=os.environ["HF_TOKEN"]
)

print(f"✅ Upload complete!")
print(f"\n🎉 Model: https://huggingface.co/{repo_id}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"CV Score: {grid_search.best_score_:.4f}")
print("="*50)
