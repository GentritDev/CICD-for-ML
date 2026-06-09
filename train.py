import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, confusion_matrix
import skops.io as sio


df = pd.read_csv("Data/tickets.csv")

df = df.rename(columns={
    "Ticket Description": "ticket_description",
    "Ticket Type": "ticket_type"
})

df = df.dropna(subset=["ticket_description", "ticket_type"])
df = df.sample(frac=1, random_state=125) # Shartimi i të dhënave

X = df["ticket_description"].values
y = df["ticket_type"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

pipe = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
    ]
)

pipe.fit(X_train, y_train)

predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print(f"Accuracy: {round(accuracy, 2) * 100}% | F1: {round(f1, 2)}")

os.makedirs("Results", exist_ok=True)

with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {accuracy.round(2)}, F1 Score = {f1.round(2)}.")

# Create and save   Confusion Matrix
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Results/model_results.png", dpi=120)

sio.dump(pipe, "Model/ticket_pipeline.skops")
