import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
import copy

# Suppress sklearn warnings about sparse outputs
warnings.filterwarnings('ignore')

class DynamicNN(nn.Module):
    """
    A simple dynamic Neural Network where the input size adapts to the feature matrix.
    """
    def __init__(self, input_size, num_classes, hidden_sizes=None):
        super(DynamicNN, self).__init__()
        
        # Adaptive Architecture: Suggestion 3
        if hidden_sizes is None:
            hidden_sizes = [max(64, input_size * 2), max(32, input_size)]
            
        layers = []
        in_dim = input_size
        
        # Build hidden layers dynamically
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(0.3))
            in_dim = h_dim
            
        # Final classification layer
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class IoTFeatureDataset(Dataset):
    """
    Dataset loader for IoT feature matrices after scikit-learn preprocessing.
    """
    def __init__(self, features, labels, uids):
        self.features = features.astype(np.float32)
        self.labels = labels
        self.uids = uids
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]), 
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.uids[idx]
        )

def build_preprocessor(df, exclude_cols=['uid', 'device'], use_power_transformer=True):
    """
    Dynamically builds a scikit-learn ColumnTransformer for the given dataframe.
    """
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    cat_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    num_cols = df[feature_cols].select_dtypes(exclude=['object']).columns.tolist()
    
    # Better Imputation and Scaling: Suggestion 4
    if use_power_transformer:
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    else:
        scaler = StandardScaler()
        
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', scaler)
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])
    
    transformers = []
    if num_cols:
        transformers.append(('num', num_pipeline, num_cols))
    if cat_cols:
        transformers.append(('cat', cat_pipeline, cat_cols))
        
    preprocessor = ColumnTransformer(transformers)
    return preprocessor

def train_and_evaluate(df_train, df_test, label_encoder, epochs=30, batch_size=64):
    """
    Trains a model for a single feature matrix and generates predictions on the test set.
    """
    # Validation Split for Early Stopping: Suggestion 5
    unique_train_uids = df_train['uid'].unique()
    if len(unique_train_uids) > 2:
        train_uids_local, val_uids_local = train_test_split(unique_train_uids, test_size=0.2, random_state=42)
        df_t = df_train[df_train['uid'].isin(train_uids_local)].copy()
        df_v = df_train[df_train['uid'].isin(val_uids_local)].copy()
    else:
        df_t = df_train.copy()
        df_v = pd.DataFrame(columns=df_train.columns)
        
    preprocessor = build_preprocessor(df_t, use_power_transformer=True)
    
    try:
        X_train = preprocessor.fit_transform(df_t)
    except Exception as e:
        print(f"    Warning: PowerTransformer failed, falling back to StandardScaler.")
        preprocessor = build_preprocessor(df_t, use_power_transformer=False)
        X_train = preprocessor.fit_transform(df_t)

    X_val = preprocessor.transform(df_v) if not df_v.empty else np.array([])
    X_test = preprocessor.transform(df_test) if not df_test.empty else np.array([])
    
    y_train = label_encoder.transform(df_t['device'].astype(str))
    y_val = label_encoder.transform(df_v['device'].astype(str)) if not df_v.empty else np.array([])
    y_test = label_encoder.transform(df_test['device'].astype(str)) if not df_test.empty else np.array([])
    
    train_dataset = IoTFeatureDataset(X_train, y_train, df_t['uid'].values)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if not df_v.empty:
        val_dataset = IoTFeatureDataset(X_val, y_val, df_v['uid'].values)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    
    model = DynamicNN(input_size=input_size, num_classes=num_classes)
    
    # Address Class Imbalance: Suggestion 1
    unique_classes = np.unique(y_train)
    if len(unique_classes) > 1:
        cw = compute_class_weight('balanced', classes=unique_classes, y=y_train)
        weights = np.ones(num_classes, dtype=np.float32)
        for idx, c in enumerate(unique_classes):
            weights[c] = cw[idx]
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Tracking for Early Stopping
    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience = 5
    epochs_no_improve = 0
    val_accuracy = 0.5 # Default fallback
    
    model.train()
    for epoch in range(epochs):
        for features, labels, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        if not df_v.empty:
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for features, labels, _ in val_loader:
                    outputs = model(features)
                    val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            current_val_acc = correct / total
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                val_accuracy = current_val_acc
                epochs_no_improve = 0
                best_weights = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    model.load_state_dict(best_weights)
                    break
            model.train()
    
    if df_v.empty:
        val_accuracy = min(1.0, len(df_t) / 1000.0) 

    # Inference on Test set
    test_predictions = {}
    if not df_test.empty:
        model.eval()
        test_dataset = IoTFeatureDataset(X_test, y_test, df_test['uid'].values)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for features, labels, batch_uids in test_loader:
                outputs = model(features)
                probs = torch.softmax(outputs, dim=1).numpy()
                for i, u in enumerate(batch_uids):
                    test_predictions[u] = probs[i]
                    
    return model, test_predictions, val_accuracy

def main():
    data_dir = 'data/processed/feature_matrices'
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not csv_files:
        print("No feature matrices found!")
        return

    print("Gathering global UIDs and Target Labels...")
    all_devices = set()
    all_uids = set()
    uid_to_device = {}
    
    for f in csv_files:
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        if 'uid' not in cols and 'uids' not in cols:
            continue
            
        usecols = ['uid', 'device'] if 'uid' in cols else ['uids', 'device']
        df = pd.read_csv(f, usecols=usecols)
        if 'uids' in df.columns:
            df = df.rename(columns={'uids': 'uid'})
            
        df['device'] = df['device'].replace('-', np.nan)
        df = df.dropna(subset=['uid', 'device'])
        
        all_devices.update(df['device'].unique().tolist())
        all_uids.update(df['uid'].unique().tolist())
        
        for _, row in df.iterrows():
            uid_to_device[row['uid']] = str(row['device'])
            
    label_encoder = LabelEncoder()
    label_encoder.fit(list(all_devices))
    print(f"Found {len(label_encoder.classes_)} unique devices.")
    print(f"Found {len(all_uids)} unique UIDs.")
    
    train_uids, test_uids = train_test_split(list(all_uids), test_size=0.2, random_state=42)
    train_uids_set = set(train_uids)
    test_uids_set = set(test_uids)
    print(f"Global Split -> Train UIDs: {len(train_uids_set)}, Test UIDs: {len(test_uids_set)}")
    
    uid_predictions = {uid: [] for uid in test_uids_set}
    
    for f in csv_files:
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        if 'uid' not in cols and 'uids' not in cols:
            continue
            
        matrix_name = os.path.basename(f).replace('_feature_matrix.csv', '')
        print(f"\nProcessing {matrix_name}...")
        df = pd.read_csv(f)
        
        if 'uids' in df.columns:
            df = df.rename(columns={'uids': 'uid'})
            
        df = df.replace('-', np.nan)
        df = df.dropna(subset=['uid', 'device'])
        
        if df.empty:
            print(f"Skipping {matrix_name}, no valid data.")
            continue
            
        for col in df.columns:
            if col not in ['uid', 'device']:
                df[col] = pd.to_numeric(df[col], errors='ignore')
                
        df_train = df[df['uid'].isin(train_uids_set)].copy()
        df_test = df[df['uid'].isin(test_uids_set)].copy()
        
        if df_train.empty:
            print(f"Skipping {matrix_name}, no training data after split.")
            continue
            
        print(f"Training on {len(df_train)} samples, testing on {len(df_test)} samples.")
        
        model, test_preds, model_weight = train_and_evaluate(df_train, df_test, label_encoder, epochs=40)
        print(f"    Model Validation Accuracy (Weight): {model_weight:.4f}")
        
        THRESHOLD = 0.30
        if model_weight < THRESHOLD:
            print(f"    -> Pruning {matrix_name} from ensemble (Validation Acc < {THRESHOLD})")
        else:
            print(f"    -> Including {matrix_name} in ensemble")
        
        # Track predictions for individual accuracy
        individual_y_true = []
        individual_y_pred = []
        
        for uid, probs in test_preds.items():
            # Only add to ensemble if the model meets the threshold
            if model_weight >= THRESHOLD:
                if uid in uid_predictions:
                    uid_predictions[uid].append((probs, model_weight))
            
            # Store labels for individual testing
            if uid in uid_to_device:
                true_label = label_encoder.transform([uid_to_device[uid]])[0]
                pred_label = np.argmax(probs)
                individual_y_true.append(true_label)
                individual_y_pred.append(pred_label)
                
        if individual_y_true:
            individual_acc = accuracy_score(individual_y_true, individual_y_pred)
            print(f"    Individual Test Accuracy: {individual_acc:.4f} ({individual_acc*100:.2f}%) on {len(individual_y_true)} test samples")
        else:
            print(f"    Individual Test Accuracy: N/A")
                
    # Weighted Ensemble Aggregation: Suggestion 2
    print("\n" + "="*40)
    print("ENSEMBLE EVALUATION (WEIGHTED)")
    print("="*40)
    
    from collections import defaultdict
    y_true = []
    y_pred_vote = []
    y_pred_avg = []
    
    evaluated_uids = 0
    for uid, predictions_list in uid_predictions.items():
        if not predictions_list:
            continue
            
        evaluated_uids += 1
        true_label = label_encoder.transform([uid_to_device[uid]])[0]
        y_true.append(true_label)
        
        # --- Option A: Weighted Majority Vote ---
        vote_scores = defaultdict(float)
        for probs, weight in predictions_list:
            vote_class = np.argmax(probs)
            vote_scores[vote_class] += weight
        majority_class = max(vote_scores.items(), key=lambda x: x[1])[0]
        y_pred_vote.append(majority_class)
        
        # --- Option B: Weighted Average Probabilities ---
        probs_array = [p for p, w in predictions_list]
        weights_array = [w for p, w in predictions_list]
        
        # Normalize weights to avoid zero sum division issues
        total_weight = sum(weights_array)
        if total_weight > 0:
            avg_probs = np.average(probs_array, axis=0, weights=weights_array)
        else:
            avg_probs = np.mean(probs_array, axis=0)
            
        avg_class = np.argmax(avg_probs)
        y_pred_avg.append(avg_class)
        
    if evaluated_uids > 0:
        acc_vote = accuracy_score(y_true, y_pred_vote)
        acc_avg = accuracy_score(y_true, y_pred_avg)
        print(f"Total Test UIDs Evaluated: {evaluated_uids}")
        print(f"Option A (Weighted Majority Vote) Accuracy: {acc_vote:.4f} ({acc_vote*100:.2f}%)")
        print(f"Option B (Weighted Average Probs) Accuracy: {acc_avg:.4f} ({acc_avg*100:.2f}%)")
        
        if acc_vote > acc_avg:
            print("Conclusion: Option A (Weighted Majority Vote) performed better.")
        elif acc_avg > acc_vote:
            print("Conclusion: Option B (Weighted Average Probabilities) performed better.")
        else:
            print("Conclusion: Both ensemble methods performed equally well.")
    else:
        print("No test predictions could be evaluated.")

if __name__ == '__main__':
    main()
