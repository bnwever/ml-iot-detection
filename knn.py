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
# 2. FEATURE SELECTION
# ─────────────────────────────────────────
X_FEATURES = [
    'L4_tcp', 'L4_udp', 'L7_http', 'L7_https', 'port_class_src',
    'port_class_dst', 'pck_size', 'ethernet_frame_size', 'ttl',
    'total_length', 'protocol', 'source_port', 'dest_port', 'DNS_count',
    'NTP_count', 'ARP_count', 'cnt', 'L3_ip_dst_count', 'most_freq_prot',
    'most_freq_sport', 'most_freq_dport', 'sum_et', 'min_et', 'max_et',
    'med_et', 'average_et', 'skew_et', 'kurt_et', 'var', 'q3', 'q1', 'iqr',
    'sum_e', 'min_e', 'max_e', 'med', 'average', 'skew_e', 'kurt_e',
    'var_e', 'q3_e', 'q1_e', 'iqr_e'
]

Y_FEATURES = ['global_category', 'device', 'interaction_type', 'command']

# Check for missing columns
missing = [col for col in X_FEATURES if col not in df.columns]
if missing:
    print(f"Warning: Missing features: {missing}")
    X_FEATURES = [col for col in X_FEATURES if col in df.columns]

print(f"\nDevices in dataset:")
print(df['device'].value_counts())

# ─────────────────────────────────────────
# 3. SEPARATE FEATURES AND TARGETS
# ─────────────────────────────────────────
X = df[X_FEATURES].copy()
y = df[Y_FEATURES].copy()

# ─────────────────────────────────────────
# 4. HANDLE NON-NUMERIC FEATURES
# ─────────────────────────────────────────
le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].fillna('unknown')
        X[col] = le.fit_transform(X[col].astype(str))
    else:
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Encode target columns
y_encoded = pd.DataFrame()
target_encoders = {}
for target in Y_FEATURES:
    enc = LabelEncoder()
    y_encoded[target] = enc.fit_transform(y[target].astype(str))
    target_encoders[target] = enc

# ─────────────────────────────────────────
# 5. HANDLE MISSING VALUES
# ─────────────────────────────────────────
X = X.fillna(0)

print(f"\nFeatures: {X.shape[1]}")
print(f"Samples: {len(X)}")

# ─────────────────────────────────────────
# 6. TRAIN/TEST SPLIT (before scaling!)
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded['device']
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ─────────────────────────────────────────
# 7. SCALE FEATURES (fit on train only!)
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# Scale full dataset for cross validation
scaler_cv = StandardScaler()
X_all_scaled = scaler_cv.fit_transform(X)
X_all_scaled = np.nan_to_num(X_all_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# ─────────────────────────────────────────
# 8. FIND BEST K USING CROSS VALIDATION
# ─────────────────────────────────────────
print("\n── Testing different values of k ──")
k_values = range(1, 16)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    scores = cross_val_score(knn, X_all_scaled, y_encoded['device'], cv=3, scoring='accuracy')
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
plt.close()
print("k tuning plot saved as knn_k_tuning.png")

# ─────────────────────────────────────────
# 9. TRAIN FINAL MODEL WITH BEST K
# ─────────────────────────────────────────
model = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
model.fit(X_train_scaled, y_train)
print(f"\nFinal model trained with k={best_k}")

# ─────────────────────────────────────────
# 10. EVALUATE ALL 4 OUTPUTS
# ─────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
y_pred_df = pd.DataFrame(y_pred, columns=Y_FEATURES)

print("\n── Classification Reports for all 4 outputs ──")
for i, target in enumerate(Y_FEATURES):
    enc = target_encoders[target]
    y_test_labels = enc.inverse_transform(y_test[target])
    y_pred_labels = enc.inverse_transform(y_pred_df[target])
    print(f"\n{'='*50}")
    print(f"Target: {target.upper()}")
    print(f"{'='*50}")
    print(classification_report(y_test_labels, y_pred_labels))

# ─────────────────────────────────────────
# 11. CONFUSION MATRIX FOR DEVICE
# ─────────────────────────────────────────
enc = target_encoders['device']
y_test_device = enc.inverse_transform(y_test['device'])
y_pred_device = enc.inverse_transform(y_pred_df['device'])

labels = sorted(enc.classes_)
cm = confusion_matrix(y_test_device, y_pred_device, labels=labels)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title(f'kNN Confusion Matrix - Device (k={best_k})')
plt.ylabel('Actual Device')
plt.xlabel('Predicted Device')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('knn_confusion_matrix.png', dpi=150)
plt.close()
print("\nConfusion matrix saved as knn_confusion_matrix.png")