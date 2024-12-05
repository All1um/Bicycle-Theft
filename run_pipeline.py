import pandas as pd
from src.data_transformation import transform_data
from src.feature_selection import correlation_analysis, feature_importance_analysis
from src.train_test_split import split_data
from src.imbalance_handling import upsample_minority, apply_smote, visualize_class_distribution
import os

def handle_class_imbalance(X_train, y_train, method='upsample'):
    """
    Handles class imbalance using the specified method.
    Available methods: 'upsample', 'smote'
    """
    if method == 'upsample':
        X_balanced, y_balanced = upsample_minority(X_train, y_train)
        print("Class imbalance handled using up-sampling.")
    elif method == 'smote':
        X_balanced, y_balanced = apply_smote(X_train, y_train)
        print("Class imbalance handled using SMOTE.")
    else:
        raise ValueError("Invalid method specified. Choose 'upsample' or 'smote'.")
    
    return X_balanced, y_balanced

def save_balanced_data(X_train_balanced, y_train_balanced):
    """
    Save the balanced training data to CSV files.
    """
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)  # Ensure the data directory exists
    X_train_path = os.path.join(data_dir, 'balanced_X_train.csv')
    y_train_path = os.path.join(data_dir, 'balanced_y_train.csv')
    
    # Save to CSV
    X_train_balanced.to_csv(X_train_path, index=False)
    y_train_balanced.to_csv(y_train_path, index=False)
    
    print(f"Balanced training data saved to '{X_train_path}' and '{y_train_path}'")

def main():
    """
    Main function to execute the data modeling pipeline.
    """
    # Data Transformation
    print("Starting data transformation...")
    transformed_data = transform_data('theft.csv')
    
    # Feature Selection
    print("\nStarting feature selection...")
    # Correlation Analysis
    corr_matrix = correlation_analysis(transformed_data)
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Feature Importance Analysis
    feature_importance_df = feature_importance_analysis(transformed_data)
    print("\nFeature Importances:")
    print(feature_importance_df)
    
    # Train-Test Split
    print("\nStarting train-test split...")
    X_train, X_test, y_train, y_test = split_data(transformed_data)
    print("\nData splitting complete.")
    
    # Managing Imbalanced Classes
    print("\nHandling class imbalance...")
    balancing_method = 'upsample'  
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train, method=balancing_method)
    
    # Save Balanced Data
    save_balanced_data(X_train_balanced, y_train_balanced)
    
    print("\nPipeline execution complete.")

if __name__ == "__main__":
    main()