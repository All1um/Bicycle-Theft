import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def upsample_minority(X_train, y_train):
    """
    Up-sample the minority class in the training data.
    """
    # Combine X_train and y_train
    y_train = y_train.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    train_data = pd.concat([X_train, y_train], axis=1)
    
    # Separate majority and minority classes
    majority_class = train_data[train_data.STATUS == 0]
    minority_class = train_data[train_data.STATUS == 1]
    
    # Check if minority_class is empty
    if len(minority_class) == 0:
        print("No minority class samples to upsample.")
        return X_train, y_train
    
    # Up-sample minority class
    minority_upsampled = resample(
        minority_class,
        replace=True,  
        n_samples=len(majority_class),  
        random_state=42
    )
    
    # Combine majority class with upsampled minority class
    upsampled_data = pd.concat([majority_class, minority_upsampled])
    
    # Shuffle the data
    upsampled_data = upsampled_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separate features and target
    X_train_balanced = upsampled_data.drop('STATUS', axis=1)
    y_train_balanced = upsampled_data['STATUS']
    
    print(f"Upsampled minority class to match majority class. New training set shape: {X_train_balanced.shape}")
    
    return X_train_balanced, y_train_balanced

def apply_smote(X_train, y_train):
    """
    Apply SMOTE to the training data.
    """
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Applied SMOTE. New training set shape: {X_train_balanced.shape}")
    return X_train_balanced, y_train_balanced

def visualize_class_distribution(y_before, y_after, title_before, title_after):
    """
    Visualize the class distribution before and after balancing.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.countplot(x=y_before)
    plt.title(title_before)
    plt.xlabel('STATUS')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.countplot(x=y_after)
    plt.title(title_after)
    plt.xlabel('STATUS')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from train_test_split import split_data
    from data_transformation import transform_data

    # Load and transform data
    data = transform_data('theft.csv')

    # Split data
    X_train, X_test, y_train, y_test = split_data(data)

    # Before balancing
    y_train_before = y_train.copy()

    # Up-sampling the minority class
    X_train_balanced, y_train_balanced = upsample_minority(X_train, y_train)

    print("\nBalanced training target distribution:\n", y_train_balanced.value_counts())