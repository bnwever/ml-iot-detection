import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib
from network import MHNetwork

def main():
    x_path = os.path.join('data', 'processed', 'X_matrix.parquet')
    y_path = os.path.join('data', 'processed', 'Y_matrix.parquet')
    models_dir = 'models'
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading datasets...")
    X_df = pd.read_parquet(x_path)
    Y_df = pd.read_parquet(y_path)

    print(f"Original X shape: {X_df.shape}, Y shape: {Y_df.shape}")

    # Handle missing values
    X_df = X_df.fillna(0)

    # Encode Y: Transform the 4 categorical columns into integers
    Y = np.zeros(Y_df.shape)
    encoders = {}
    for i, col in enumerate(Y_df.columns):
        le = LabelEncoder()
        Y[:, i] = le.fit_transform(Y_df[col].astype(str))
        encoders[col] = le
        
    joblib.dump(encoders, os.path.join(models_dir, 'encoders_multi.pkl'))
    print("Saved LabelEncoders to models/encoders_multi.pkl")

    # Split the dataset 70/30
    print("Splitting data into 70% training and 30% testing...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_df.values, Y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    
    # Save the scaler for use in the testing script
    joblib.dump(scaler, os.path.join(models_dir, 'scaler_multi.pkl'))
    print("Saved StandardScaler to models/scaler_multi.pkl")

    # Convert training features to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # Architecture definitions
    input_size = X_train.shape[1] 
    hidden_size = 24
    output_sizes = [len(encoders[col].classes_) for col in Y_df.columns]
    
    print(f"\nTraining multi-headed model...")
    print(f"Inputs={input_size}, Hidden={hidden_size}, Output Heads={output_sizes}")
    epochs = 100

    model = MHNetwork(input_size, hidden_size, output_sizes)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train
    for epoch in range(epochs):
        model.train()
        
        outputs = model(X_train_tensor)
        
        loss = 0
        for i in range(len(output_sizes)):
            target = y_train_tensor[:, i]
            loss += criterion(outputs[i], target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Save the model
    model_path = os.path.join(models_dir, f"multi_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained multi-headed model to {model_path}")
    
    print("\nTraining completed. Run test_multi_nn.py to evaluate the model on the test set.")

if __name__ == "__main__":
    main()
