import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================
# PATHS
# ==============================

DATA_FILE = "data/temporal/activity_temporal_dataset.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

MODEL_FILE = "models/temporal_activity_classifier.joblib"
CONFUSION_FILE = "outputs/temporal_activity_confusion_matrix.png"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# LOAD DATA
# ==============================

print("==========================================")
print("FOCUSLENS AI - ACTIVITY MODEL V2")
print("==========================================")

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}")

# Remove missing rows
df = df.dropna()

print(f"Shape after removing missing values: {df.shape}")


# ==============================
# CLASS DISTRIBUTION
# ==============================

print("\nClass distribution:")
print(df["label"].value_counts())


# ==============================
# FEATURES / LABEL
# ==============================

X = df.drop(columns=["label"])
y = df["label"]

print(f"\nNumber of features: {X.shape[1]}")


# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# RANDOM FOREST MODEL
# ==============================

print("\nTraining Random Forest V2...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ==============================
# PREDICTION
# ==============================

y_pred = model.predict(X_test)


# ==============================
# ACCURACY
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("MODEL RESULTS")
print("==========================================")

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")


# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("FocusLens AI - Temporal Activity Model V2")

plt.tight_layout()

plt.savefig(
    CONFUSION_FILE,
    dpi=300
)

plt.close()


# ==============================
# FEATURE IMPORTANCE
# ==============================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": importance
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 15 Important Features:")

print(
    feature_importance.head(15).to_string(
        index=False
    )
)


# ==============================
# SAVE MODEL
# ==============================

joblib.dump(
    model,
    MODEL_FILE
)

print("\n==========================================")
print("TRAINING COMPLETE")
print("==========================================")

print(f"\nModel saved at:")
print(MODEL_FILE)

print("\nConfusion matrix saved at:")
print(CONFUSION_FILE)