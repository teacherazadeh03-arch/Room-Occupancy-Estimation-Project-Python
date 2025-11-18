# ========================================================
# Room Occupancy Estimation — Machine Learning Models
# Dataset: UCI Room Occupancy Estimation (ID = 864) :contentReference[oaicite:0]{index=0}
# ========================================================

# 1) Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# 2) Load Dataset via ucimlrepo
%pip install ucimlrepo

from ucimlrepo import fetch_ucirepo

room = fetch_ucirepo(id=864)
X = room.data.features
y = room.data.targets

# If target is a DataFrame, convert to Series
if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
    y = y.iloc[:, 0]

print("Shape:", X.shape, y.shape)
display(X.head())
print("Target distribution:", y.value_counts())

# 3) Preprocess Data
# Drop non-numeric columns (Date and Time)
X_numeric = X.drop(columns=["Date", "Time"])

# Optional: check remaining columns
print(X_numeric.head())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# 4) Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 5) K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))

# 6) Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Report:\n", classification_report(y_test, y_pred_lr))

# 7) Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Decision Tree Report:\n", classification_report(y_test, y_pred_dt))

# 8) Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))

# 9) Hierarchical Clustering (on the full dataset, unsupervised)
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title("Hierarchical Clustering Dendrogram (Ward)")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Cut the dendrogram to create clusters; let's say 4 clusters
clusters = fcluster(linked, t=4, criterion='maxclust')
print("Cluster counts:", pd.Series(clusters).value_counts())

# Optionally, evaluate how clusters align with true occupancy
df_clusters = pd.DataFrame({"cluster": clusters, "occupancy": y.values})
print(pd.crosstab(df_clusters["cluster"], df_clusters["occupancy"]))

# 10) Confusion Matrices
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

plot_conf_matrix(y_test, y_pred_knn, "KNN Confusion Matrix")
plot_conf_matrix(y_test, y_pred_lr, "Logistic Regression Confusion Matrix")
plot_conf_matrix(y_test, y_pred_dt, "Decision Tree Confusion Matrix")
plot_conf_matrix(y_test, y_pred_rf, "Random Forest Confusion Matrix")

# 11) ROC-AUC (if meaningful)
# If target is multi-class, this is less straightforward; if it's binary, you can do:
try:
    y_prob_knn = knn.predict_proba(X_test)[:, 1]
    print("KNN ROC-AUC:", roc_auc_score(y_test, y_prob_knn))
except Exception as e:
    print("Couldn't compute ROC-AUC for KNN:", e)

try:
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, y_prob_lr))
except Exception as e:
    print("Couldn't compute ROC-AUC for LR:", e)

try:
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    print("Random Forest ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
except Exception as e:
    print("Couldn't compute ROC-AUC for RF:", e)

# 12) Feature importance for Random Forest
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=importances.index)
plt.title("Random Forest Feature Importance")
plt.show()

print("Done.")
