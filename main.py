import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv('IrisFlowerClassification/Data/Iris.csv')

# Clean and prepare data
df.drop(columns=['Id'], inplace=True)
X = df.drop(columns=['Species'])
y = df['Species']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

# Visualizations
sns.set_style("darkgrid")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("🌸 Iris Flower Classification - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("🔍 Classification Report:\n")
print(report)

# Visualizing the decision boundaries using PCA (just for 2D plotting)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 7))
palette = ['#FF6663', '#FEB144', '#9EE09E']
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Species'], palette=palette, s=100, edgecolor='black')
plt.title("🌸 PCA Projection of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title='Species')
plt.show()
