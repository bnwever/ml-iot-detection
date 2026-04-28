import os
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from train_nn_ciciot import SimpleNN, CICIoTDataset

def main():
    print("Loading data...")
    df = pd.read_parquet('data/processed/ciciot2022.parquet')
    
    # Remove uniquely identifying values/features and alternative labels
    drop_cols = ['ip_dst_new', 'most_freq_d_ip', 'epoch_timestamp', 
                 'global_category', 'interaction_type', 'command']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # Target classification
    target_col = 'device'
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # We reproduce the split so we test on the same 20% holdout set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Loading preprocessing artifacts...")
    model_dir = 'models/trained/ciciot'
    le = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    train_columns = joblib.load(os.path.join(model_dir, 'features.joblib'))
    
    y_test_encoded = le.transform(y_test)
    num_classes = len(le.classes_)
    
    # Process testing data
    for col in X_test.columns:
        if X_test[col].dtype == 'bool':
            X_test[col] = X_test[col].astype(int)
            
    cat_cols = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        X_test = pd.get_dummies(X_test, columns=cat_cols)
        
    X_test = X_test.fillna(0)
    
    # Reindex columns to match the training set exactly
    X_test = X_test.reindex(columns=train_columns, fill_value=0)
    
    X_test_scaled = scaler.transform(X_test)
    
    test_dataset = CICIoTDataset(X_test_scaled, y_test_encoded)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    input_dim = X_test_scaled.shape[1]
    model = SimpleNN(input_dim, num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    
    print("Evaluating model...")
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report (Test Set):")
    print(classification_report(all_targets, all_preds, target_names=le.classes_))

if __name__ == '__main__':
    main()
