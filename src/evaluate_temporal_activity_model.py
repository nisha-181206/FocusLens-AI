import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ==========================================
# PATHS
# ==========================================

DATASET_PATH = "data/temporal/activity_temporal_dataset.csv"
MODEL_PATH = "models/temporal_activity_classifier.joblib"
OUTPUT_PATH = "outputs/temporal_activity_confusion_matrix.png"


# ==========================================
# LOAD DATASET
# ==========================================

print("=" * 60)
print("FOCUSLENS AI - MODEL EVALUATION")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print("\nDataset shape:", df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())


# ==========================================
# PREPARE FEATURES AND LABELS
# ==========================================

X = df.drop(columns=["label"])
y = df["label"]


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print("\nLoading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL ACCURACY")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\nLabels:")
print(model.classes_)

print("\n")
print(cm)


# ==========================================
# SAVE CONFUSION MATRIX
# ==========================================

os.makedirs("outputs", exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

fig, ax = plt.subplots(figsize=(8, 6))

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title("FocusLens AI - Temporal Activity Classifier")
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nConfusion matrix saved to:")
print(OUTPUT_PATH)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Test samples: {len(y_test)}")
print(f"Number of classes: {len(model.classes_)}")

print("\nClasses:")

for class_name in model.classes_:
    print(f" - {class_name}")

print("\nEvaluation finished successfully.")