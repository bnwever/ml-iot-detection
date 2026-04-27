import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_parquet("ciciot2022.parquet")

print("Dataset shape:", df.shape)
print("\nDevices in dataset:")
print(df['device'].value_counts())


# ─────────────────────────────────────────
# 2. SELECT FEATURES
# ─────────────────────────────────────────
# These are the most useful behavioral features for IoT fingerprinting
# (protocol, timing stats, packet size stats)
features = [
    # Protocol flags
    'L4_tcp', 'L4_udp', 'L7_http', 'L7_https',

    # Packet size stats
    'pck_size', 'ethernet_frame_size', 'total_length',

    # Inter-arrival time stats (captures timing behavior)
    'sum_et', 'min_et', 'max_et', 'med_et', 'average_et',
    'skew_et', 'kurt_et',

    # Packet energy/size distribution stats
    'sum_e', 'min_e', 'max_e', 'med', 'average',
    'skew_e', 'kurt_e',

    # Port and protocol info
    'source_port', 'dest_port',

    # Count-based features
    'DNS_count', 'NTP_count', 'ARP_count', 'cnt',

    # Inter-arrival time
    'inter_arrival_time',
]

target = 'device'

# ─────────────────────────────────────────
# 3. PREPARE DATA
# ─────────────────────────────────────────
# Drop rows where any selected feature or target is missing
df_clean = df[features + [target]].dropna()

print(f"\nRows after dropping nulls: {len(df_clean)}")

X = df_clean[features]
y = df_clean[target]

# Convert boolean columns to int (True/False -> 1/0)
X = X.replace({True: 1, False: 0})
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# ─────────────────────────────────────────
# 4. SCALE FEATURES
# ─────────────────────────────────────────
# IMPORTANT: kNN relies on distance, so scaling is critical
# Without this, features like 'sum_et' (large numbers) will 
# dominate over boolean features like 'L4_tcp'
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────
# 5. TRAIN/TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,      # 80% train, 20% test
    random_state=42,    # for reproducibility
    stratify=y          # ensures each device is represented in both splits
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ─────────────────────────────────────────
# 6. TRAIN kNN MODEL
# ─────────────────────────────────────────
# n_neighbors=5 is a common starting point
# we'll tune this later
k = 5
model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
model.fit(X_train, y_train)

print(f"\nModel trained with k={k}")

# ─────────────────────────────────────────
# 7. EVALUATE
# ─────────────────────────────────────────
y_pred = model.predict(X_test)

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────
# 8. CONFUSION MATRIX PLOT
# ─────────────────────────────────────────
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title(f'kNN Confusion Matrix (k={k})')
plt.ylabel('Actual Device')
plt.xlabel('Predicted Device')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('knn_confusion_matrix.png', dpi=150)
plt.show()
print("\nConfusion matrix saved as knn_confusion_matrix.png")

# ─────────────────────────────────────────
# 9. FIND BEST K (optional but impressive)
# ─────────────────────────────────────────
print("\n── Testing different values of k ──")
k_values = range(1, 16)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    accuracies.append(acc)
    print(f"k={k:2d}  accuracy={acc:.4f}")

best_k = k_values[accuracies.index(max(accuracies))]
print(f"\nBest k: {best_k} with accuracy: {max(accuracies):.4f}")

# Plot k vs accuracy
plt.figure(figsize=(8, 4))
plt.plot(k_values, accuracies, marker='o', color='steelblue')
plt.axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
plt.title('kNN Accuracy vs k')
plt.xlabel('k (number of neighbors)')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('knn_k_tuning.png', dpi=150)
plt.show()
print("k tuning plot saved as knn_k_tuning.png")

