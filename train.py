import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # Use joblib instead of pickle
import os

# Load data
df = pd.read_csv("Titanic-Dataset.csv")

# Drop unnecessary columns
X = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Survived"], axis=1)
y = df["Survived"]

# Separate numeric and categorical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

clf = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
clf.fit(X_train, y_train)

# Validate
preds = clf.predict(X_valid)
acc = accuracy_score(y_valid, preds)
print(f"Validation Accuracy: {acc:.4f}")

# Create directory if it doesn't exist
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Save model using joblib
model_path = os.path.join(model_dir, "model.pkl")
joblib.dump(clf, model_path)
print(f"Model saved to: {model_path}")
