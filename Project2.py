import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

iris_data = load_iris()

dataset = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
dataset["target"] = iris_data.target
class_names = iris_data.target_names

print("Dataset shape:", dataset.shape)
print("\nSample rows:")
print(dataset.sample(5, random_state=1))
print("\nColumn info:")
print(dataset.info())
print("\nNull value check:")
print(dataset.isnull().sum())
print("\nStatistical summary:")
print(dataset.describe())
print("\nClass distribution:")
print(dataset["target"].value_counts())

features = dataset.drop(columns=["target"])
labels = dataset["target"]

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.25, random_state=7, stratify=labels
)

print("\nTrain set size:", features_train.shape[0])
print("Test set size:", features_test.shape[0])

feature_scaler = StandardScaler()
features_train_scaled = feature_scaler.fit_transform(features_train)
features_test_scaled = feature_scaler.transform(features_test)

knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(features_train_scaled, labels_train)

label_predictions = knn_classifier.predict(features_test_scaled)

print("\nPredicted labels:", list(label_predictions))
print("True labels:     ", list(labels_test.values))

print("\nDetailed classification report:")
print(classification_report(labels_test, label_predictions, target_names=class_names))

macro_f1 = f1_score(labels_test, label_predictions, average="macro")
weighted_f1 = f1_score(labels_test, label_predictions, average="weighted")
print("Macro F1 score:", round(macro_f1, 4))
print("Weighted F1 score:", round(weighted_f1, 4))

conf_matrix = confusion_matrix(labels_test, label_predictions)

plt.figure(figsize=(7, 5))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Species")
plt.ylabel("True Species")
plt.title("Iris KNN Classifier - Confusion Matrix")
plt.tight_layout()
plt.savefig("irisConfusionMatrix.png")

print("\nSaved confusion matrix")
