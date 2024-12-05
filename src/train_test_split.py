import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """
    X = data.drop('STATUS', axis=1)
    y = data['STATUS']
    
    # Ensure y is integer type
    y = y.astype(int)
    
    # Split the dataset (80% training, 20% testing) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    print("Training features shape:", X_train.shape)
    print("Testing features shape:", X_test.shape)
    print("\nTraining target distribution:\n", y_train.value_counts())
    print("\nTesting target distribution:\n", y_test.value_counts())
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from data_transformation import transform_data
    data = transform_data('theft.csv')
    X_train, X_test, y_train, y_test = split_data(data)