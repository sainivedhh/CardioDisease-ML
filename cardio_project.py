import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv('cardio_train.csv', sep=';')
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Basic info
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Visualizations
df.hist(figsize=(15,10))
plt.suptitle('Feature Distributions')
plt.show()

plt.figure(figsize=(15,10))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 4, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
for col in categorical_features:
    sns.countplot(x=col, hue='cardio', data=df)
    plt.title(f'{col} vs Target')
    plt.show()

# Preprocessing
X = df.drop('cardio', axis=1)
y = df['cardio']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Models
models = {
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

print("\nModel Accuracy Comparison:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# Save best model (you can remove if >25MB)
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
joblib.dump(best_model, 'heart_disease_model.pkl')
print(f"Saved best model: {best_model_name}")
