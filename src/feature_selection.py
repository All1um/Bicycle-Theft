import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def correlation_analysis(data):
    """
    Perform correlation analysis.
    Returns the correlation matrix.
    """
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Compute correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Return the correlation matrix
    return corr_matrix

def feature_importance_analysis(data):
    """
    Perform feature importance analysis using Random Forest.
    Returns a DataFrame of feature importances.
    """
    # Separate features and target
    X = data.drop('STATUS', axis=1)
    y = data['STATUS']
    
    # Ensure X has only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Debug: Print data types of X
    print("\nData types of features:")
    print(X.dtypes)
    
    # Initialize RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)
    
    # Fit the model
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Return the DataFrame
    return feature_importance_df

if __name__ == "__main__":
    from data_transformation import transform_data
    data = transform_data('theft.csv')
    
    # Correlation analysis
    print("\nPerforming correlation analysis...")
    corr_matrix = correlation_analysis(data)
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Feature importance analysis
    print("\nPerforming feature importance analysis...")
    feature_importance_df = feature_importance_analysis(data)
    print("\nFeature Importances:")
    print(feature_importance_df)