import os
import glob
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def train_rf_models(data_dir: str, models_dir: str):
    """
    Trains and optimizes a Random Forest model for each feature matrix.
    Saves the best trained models to disk.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    matrix_files = glob.glob(os.path.join(data_dir, '*_feature_matrix.csv'))
    if not matrix_files:
        print(f"No feature matrices found in {data_dir}")
        return
    
    trained_models = {}
    
    # Define a grid of hyperparameters to search over
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
    
    for file_path in matrix_files:
        log_type = os.path.basename(file_path).replace('_feature_matrix.csv', '')
        print(f"=== Training & Optimizing Random Forest for {log_type.upper()} ===")
        
        df = pd.read_csv(file_path)
        
        if 'device' not in df.columns:
            print(f"Skipping {log_type}: 'device' target column not found.\n")
            continue
            
        if len(df) < 50:
            print(f"Skipping {log_type}: Insufficient data ({len(df)} rows). Requires at least 50 to train effectively.\n")
            continue
            
        y = df['device']
        
        # We completely ignore uid for training
        X = df.drop(columns=['device'])
        if 'uid' in X.columns:
            X = X.drop(columns=['uid'])
            
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        numerical_transformer = SimpleImputer(strategy='median')
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
            
        rf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'))
        ])
        
        # 2. Keep it random (Standard train/test split, ignoring UID completely)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Grid Search Optimization
        print(f"Running Grid Search for {log_type} on {len(X_train)} training samples...")
        # cv=3 means 3-fold cross-validation during the grid search
        # We use KFold instead of StratifiedKFold to avoid errors when classes have < 3 samples
        cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(rf_pipeline, param_grid, cv=cv_strategy, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        print(f"Best Hyperparameters: {grid_search.best_params_}\n")
        
        # 3. Saving the model
        model_save_path = os.path.join(models_dir, f'rf_{log_type}_model.joblib')
        joblib.dump(best_model, model_save_path)
        
        trained_models[log_type] = best_model
        
    print(f"All optimized models saved to: {models_dir}")
    return trained_models

if __name__ == "__main__":
    matrices_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'feature_matrices')
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'trained', 'random_forest')
    
    train_rf_models(matrices_dir, save_dir)
