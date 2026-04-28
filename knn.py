import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_parquet("data/processed/ciciot2022.parquet")

print(f"Dataset shape: {df.shape}")

# ─────────────────────────────────────────
# 2. SEPARATE FEATURES AND TARGET
# ─────────────────────────────────────────
# Drop target, metadata, and IP columns (IP = data leakage)
drop_cols = [
    'device',           # target
    'global_category',  # metadata
    'interaction_type', # metadata
    'command',          # metadata
    'epoch_timestamp',  # timestamp
    'ip_dst_new',       # IP leakage
    'most_freq_d_ip',   # IP leakage
]

y = df['device']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])

print(f"\nDevices in dataset:")
print(y.value_counts())

# ─────────────────────────────────────────
# 3. HANDLE NON-NUMERIC FEATURES
# ─────────────────────────────────────────
le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].fillna('unknown')
        X[col] = le.fit_transform(X[col].astype(str))
    else:
        X[col] = pd.to_numeric(X[col], errors='coerce')

# ─────────────────────────────────────────
# 4. HANDLE MISSING VALUES
# ─────────────────────────────────────────
X = X.dropna(axis=1, how='all')
X = X.fillna(0)

print(f"\nFeatures after cleaning: {X.shape[1]}")

# ─────────────────────────────────────────
# 5. SCALE FEATURES
# ─────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# ─────────────────────────────────────────
# 6. REMOVE RARE CLASSES
# ─────────────────────────────────────────
min_samples = 3
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= min_samples].index
removed = class_counts[class_counts < min_samples].index.tolist()
if removed:
    print(f"\nRemoving rare classes: {removed}")

mask = y.isin(valid_classes)
X_scaled = X_scaled[mask.values]
y = y[mask]

print(f"Remaining samples: {len(y)}")

# ─────────────────────────────────────────
# 7. TRAIN/TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ─────────────────────────────────────────
# 8. FIND BEST K
# ─────────────────────────────────────────
print("\n── Testing different values of k ──")
k_values = range(1, 16)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    scores = cross_val_score(knn, X_scaled, y, cv=3, scoring='accuracy')
    acc = scores.mean()
    accuracies.append(acc)
    print(f"k={k:2d}  accuracy={acc:.4f}  (+/- {scores.std():.4f})")

best_k = list(k_values)[accuracies.index(max(accuracies))]
print(f"\nBest k: {best_k} with accuracy: {max(accuracies):.4f}")

# Plot k vs accuracy
plt.figure(figsize=(8, 4))
plt.plot(k_values, accuracies, marker='o', color='steelblue')
plt.axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
plt.title('kNN Accuracy vs k (3-fold Cross Validation)')
plt.xlabel('k (number of neighbors)')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('knn_k_tuning.png', dpi=150)
plt.show()
print("k tuning plot saved as knn_k_tuning.png")

# ─────────────────────────────────────────
# 9. TRAIN FINAL MODEL WITH BEST K
# ─────────────────────────────────────────
model = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
model.fit(X_train, y_train)
print(f"\nFinal model trained with k={best_k}")

# ─────────────────────────────────────────
# 10. EVALUATE
# ─────────────────────────────────────────
y_pred = model.predict(X_test)

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────
# 11. CONFUSION MATRIX
# ─────────────────────────────────────────
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title(f'kNN Confusion Matrix (k={best_k})')
plt.ylabel('Actual Device')
plt.xlabel('Predicted Device')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('knn_confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved as knn_confusion_matrix.png")
