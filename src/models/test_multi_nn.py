import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import os
import joblib
from network import MHNetwork

def main():
    x_path = os.path.join('data', 'processed', 'X_matrix.parquet')
    y_path = os.path.join('data', 'processed', 'Y_matrix.parquet')
    models_dir = 'models'
    
    print("Loading datasets...")
    X_df = pd.read_parquet(x_path)
    Y_df = pd.read_parquet(y_path)
    X_df = X_df.fillna(0)
    
    # Load the scaler and transform X
    scaler_path = os.path.join(models_dir, 'scaler_multi.pkl')
    if not os.path.exists(scaler_path):
        print(f"Error: {scaler_path} not found. Run train_multi_nn.py first.")
        return
        
    scaler = joblib.load(scaler_path)
    X = scaler.transform(X_df.values)
    print("Loaded StandardScaler and transformed inputs.")

    # Load the encoders and transform Y
    encoders_path = os.path.join(models_dir, 'encoders_multi.pkl')
    if not os.path.exists(encoders_path):
        print(f"Error: {encoders_path} not found. Have you run train_multi_nn.py yet?")
        return
        
    encoders = joblib.load(encoders_path)
    Y = np.zeros(Y_df.shape)
    for i, col in enumerate(Y_df.columns):
        Y[:, i] = encoders[col].transform(Y_df[col].astype(str))
    print("Loaded LabelEncoders and transformed targets.")

    # Split the dataset 70/30
    print("Splitting data to extract test set...")
    _, X_test, _, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Convert to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    input_size = X_test.shape[1] 
    hidden_size = 24
    output_sizes = [len(encoders[col].classes_) for col in Y_df.columns]
    
    print(f"\nEvaluating multi-headed model...")
    
    # Load model
    model = MHNetwork(input_size, hidden_size, output_sizes)
    model_path = os.path.join(models_dir, "multi_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Skipping...")
        return
        
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    evaluation_metrics = []
    criterion = nn.NLLLoss()

    # Test
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        
        for i, col in enumerate(Y_df.columns):
            print(f"\n=========================================")
            print(f"Evaluating '{col}' ({output_sizes[i]} classes)")
            print(f"=========================================")
            
            target = y_test_tensor[:, i]
            output = test_outputs[i]
            
            test_loss = criterion(output, target)
            print(f"Test NLL Loss: {test_loss.item():.4f}")
            
            # Convert log probabilities to predicted classes
            _, predicted = torch.max(output.data, 1)
            
            target_np = target.numpy()
            predicted_np = predicted.numpy()
            
            # Calculate accuracy
            acc = accuracy_score(target_np, predicted_np)
            print(f"Accuracy for {col}: {acc:.4f}")
            
            target_names = [str(c) for c in encoders[col].classes_]
            
            print(f"\nClassification Report for {col}:")
            
            from sklearn.metrics import confusion_matrix
            report_dict = classification_report(target_np, predicted_np, target_names=target_names, zero_division=0, output_dict=True)
            cm = confusion_matrix(target_np, predicted_np)
            total = np.sum(cm)
            
            overall_acc = report_dict.pop('accuracy')
            df_report = pd.DataFrame(report_dict).transpose()
            
            acc_column = []
            for idx in df_report.index:
                if idx in target_names:
                    idx_i = target_names.index(idx)
                    TP = cm[idx_i, idx_i]
                    FP = np.sum(cm[:, idx_i]) - TP
                    FN = np.sum(cm[idx_i, :]) - TP
                    TN = total - TP - FN - FP
                    class_acc = (TP + TN) / total if total > 0 else 0
                    acc_column.append(class_acc)
                elif idx == 'macro avg':
                    acc_column.append(np.mean([acc_column[j] for j in range(len(target_names))]))
                elif idx == 'weighted avg':
                    # Use the overall model accuracy
                    acc_column.append(overall_acc)
                else:
                    acc_column.append(overall_acc)
            
            df_report.insert(0, 'accuracy', acc_column)
            
            # Format support as integer
            df_report['support'] = df_report['support'].astype(int)
            print(df_report.to_string(float_format=lambda x: f"{x:.4f}"))
            
            # Extract weighted average metrics for the final summary
            precision, recall, fscore, _ = precision_recall_fscore_support(
                target_np, predicted_np, average='weighted', zero_division=0
            )
            
            evaluation_metrics.append({
                "Target": col,
                "Accuracy": round(acc, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-Score": round(fscore, 4)
            })
            
    if evaluation_metrics:
        print("\n--- Final Evaluation Summary ---")
        summary_df = pd.DataFrame(evaluation_metrics)
        summary_df.set_index("Target", inplace=True)
        print(summary_df.to_string())
        
        avg_acc = summary_df["Accuracy"].mean()
        avg_prec = summary_df["Precision"].mean()
        avg_rec = summary_df["Recall"].mean()
        avg_f1 = summary_df["F1-Score"].mean()
        print(f"\nAverage Accuracy: {avg_acc:.4f}")
        print(f"Average Precision: {avg_prec:.4f}")
        print(f"Average Recall: {avg_rec:.4f}")
        print(f"Average F1-Score: {avg_f1:.4f}")

if __name__ == "__main__":
    main()
