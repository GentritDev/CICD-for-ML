import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score,
    ConfusionMatrixDisplay, confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import skops.io as sio

# Load and clean
df = pd.read_csv("Data/tickets.csv")
df = df.rename(columns={
    "Ticket Description": "ticket_description",
    "Ticket Type": "ticket_type"
})
df = df.dropna(subset=["ticket_description", "ticket_type"])

# Log class distribution
print("Class distribution:")
print(df["ticket_type"].value_counts())
print()

X = df["ticket_description"].values
y = df["ticket_type"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125, stratify=y
)


pipe = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),      
            sublinear_tf=True        
        )),
        ("model", LinearSVC(
            random_state=125,
            class_weight="balanced", 
            max_iter=2000
        )),
    ]
)

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print(f"Accuracy: {round(accuracy * 100, 1)}% | Macro F1: {round(f1, 3)}")
print()

report = classification_report(y_test, predictions)
print(report)

os.makedirs("Results", exist_ok=True)

with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy = {accuracy:.2f}, F1 Score = {f1:.2f}\n\n")
    outfile.write("Per-class metrics:\n")
    outfile.write(report)

cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot(ax=ax)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("Results/model_results.png", dpi=120)

sio.dump(pipe, "Model/ticket_pipeline.skops")
