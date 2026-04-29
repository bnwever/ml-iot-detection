import os
import glob
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def test_rf_models(data_dir: str, models_dir: str):
    """
    Loads trained Random Forest models and evaluates them on the test set.
    """
    matrix_files = glob.glob(os.path.join(data_dir, '*_feature_matrix.csv'))
    if not matrix_files:
        print(f"No feature matrices found in {data_dir}")
        return
    
    total_correct = 0
    total_samples = 0
    
    for file_path in matrix_files:
        log_type = os.path.basename(file_path).replace('_feature_matrix.csv', '')
        model_path = os.path.join(models_dir, f'rf_{log_type}_model.joblib')
        
        if not os.path.exists(model_path):
            print(f"Skipping {log_type}: Model file not found at {model_path}\n")
            continue
            
        print(f"=== Evaluating Random Forest for {log_type.upper()} ===")
        
        df = pd.read_csv(file_path)
        
        if 'device' not in df.columns:
            print(f"Skipping {log_type}: 'device' target column not found.\n")
            continue
            
        if len(df) < 50:
            print(f"Skipping {log_type}: Insufficient data ({len(df)} rows).\n")
            continue
            
        y = df['device']
        
        X = df.drop(columns=['device'])
        if 'uid' in X.columns:
            X = X.drop(columns=['uid'])
            
        # Standard train/test split, must match the random_state used in training
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        
        print(f"Evaluating on {len(X_test)} test samples...")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {log_type}: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("\n")
        
        correct_predictions = sum(y_test == y_pred)
        total_correct += correct_predictions
        total_samples += len(y_test)
        
    if total_samples > 0:
        overall_accuracy = total_correct / total_samples
        print(f"==================================================")
        print(f"OVERALL COMBINED TEST ACCURACY: {overall_accuracy:.4f} ({total_correct}/{total_samples} correct)")
        print(f"==================================================\n")

if __name__ == "__main__":
    matrices_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'feature_matrices')
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'trained', 'random_forest')
    
    test_rf_models(matrices_dir, models_dir)
