# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
# for hugging face space authentication to upload files
from huggingface_hub import HfApi
from datetime import datetime

# Define constants for the dataset and output paths
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
DATASET_PATH = "hf://datasets/Hugo014/Tourism-Package-Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("✓ Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)
print("✓ CustomerID column dropped.")

# Treat the Gender column
print("\n--- Fixing Gender column ---")
print(f"Before: {df['Gender'].unique()}")
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
print(f"After: {df['Gender'].unique()}")

# Encoding Categorical cols.
print("\n--- Encoding categorical features ---")

# Label Encoding for nominal features
label_encode_cols = ['Gender', 'TypeofContact', 'Occupation', 'MaritalStatus']

for col in label_encode_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    print(f"✓ {col} encoded: {df[col].unique()}")

# Ordinal Encoding for hierarchical features
print("\n--- Ordinal encoding ---")

product_order = [['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King']]
ord_enc_product = OrdinalEncoder(categories=product_order)
df['ProductPitched'] = ord_enc_product.fit_transform(df[['ProductPitched']])
print(f"✓ ProductPitched encoded: {df['ProductPitched'].unique()}")

designation_order = [['Manager', 'Senior Manager', 'AVP', 'VP', 'Executive']]
ord_enc_des = OrdinalEncoder(categories=designation_order)
df['Designation'] = ord_enc_des.fit_transform(df[['Designation']])
print(f"✓ Designation encoded: {df['Designation'].unique()}")

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\n--- Train-test split ---")
# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✓ Training set: {Xtrain.shape}")
print(f"✓ Test set: {Xtest.shape}")

# Save files locally
print("\n--- Saving files locally ---")
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("✓ All files saved locally")

# Upload to Hugging Face with unique commit message
print("\n--- Uploading to Hugging Face ---")
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Create unique commit message with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
commit_msg = f"Update preprocessed data - {timestamp}"

for file_path in files:
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path,
            repo_id="Hugo014/Tourism-Package-Prediction",
            repo_type="dataset",
            token=os.environ["HF_TOKEN"],
            commit_message=commit_msg  # ← Force new commit
        )
        print(f"✓ Uploaded: {file_path}")
    except Exception as e:
        print(f"❌ Failed to upload {file_path}: {e}")

print("\n🎉 Preprocessing complete!")
print(f"Commit message: {commit_msg}")
print("View at: https://huggingface.co/datasets/Hugo014/Tourism-Package-Prediction")
