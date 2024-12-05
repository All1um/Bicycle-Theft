import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_data(filename):
    """
    Load the dataset from a CSV file using a dynamically resolved path.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', filename)
    data = pd.read_csv(data_path)
    print(f"Loaded data with shape: {data.shape}")
    return data

def handle_missing_data(data):
    """
    Handle missing data by dropping columns with significant missing values
    and imputing others.
    """
    # Drop columns with significant missing values
    columns_to_drop = ['BIKE_MODEL', 'BIKE_SPEED', 'BIKE_COLOUR']
    data = data.drop(columns=columns_to_drop, errors='ignore')
    print(f"\nDropped columns with significant missing values: {columns_to_drop}")
    
    # Fill missing values in numerical columns with mean
    numerical_cols = ['BIKE_COST']
    for col in numerical_cols:
        if col in data.columns:
            mean_value = data[col].mean()
            data[col] = data[col].fillna(mean_value)
            print(f"Filled missing values in numerical column '{col}' with mean: {mean_value:.2f}")
    
    # Fill missing values in categorical columns with mode
    categorical_cols = ['BIKE_MAKE', 'BIKE_TYPE']
    for col in categorical_cols:
        if col in data.columns:
            mode_value = data[col].mode()[0]
            data[col] = data[col].fillna(mode_value)
            print(f"Filled missing values in categorical column '{col}' with mode: {mode_value}")
    
    return data

def encode_categorical_data(data):
    """
    Encode categorical columns using pandas' Categorical type.
    Returns the encoded DataFrame and a dictionary of category mappings.
    """
    # Drop unique identifier columns
    columns_to_drop = ['OBJECTID', 'EVENT_UNIQUE_ID']
    data = data.drop(columns=columns_to_drop, errors='ignore')
    print(f"\nDropped unique identifier columns: {columns_to_drop}")
    
    # Define categorical columns to encode
    categorical_cols = ['PRIMARY_OFFENCE', 'DIVISION', 'LOCATION_TYPE', 'PREMISES_TYPE',
                        'HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140',
                        'BIKE_MAKE', 'BIKE_TYPE']
    
    # Encode target variable 'STATUS'
    status_mapping = {'STOLEN': 0, 'RECOVERED': 1}
    data['STATUS'] = data['STATUS'].map(status_mapping)
    
    # Drop rows where 'STATUS' is NaN after mapping
    initial_rows = data.shape[0]
    data = data.dropna(subset=['STATUS'])
    dropped_rows = initial_rows - data.shape[0]
    print(f"After mapping 'STATUS', dropped {dropped_rows} rows with NaN in 'STATUS'")
    
    # Initialize a dictionary to store mappings (optional, useful for future reference)
    category_mappings = {}
    
    # Encode other categorical columns using pandas' Categorical type
    for col in categorical_cols:
        if col in data.columns:
            data[col] = pd.Categorical(data[col])
            category_mappings[col] = dict(enumerate(data[col].cat.categories))
            data[col] = data[col].cat.codes
            print(f"Encoded categorical column '{col}'")
    
    # Explicitly cast 'STATUS' to integer type
    data['STATUS'] = data['STATUS'].astype(int)
    print("Cast 'STATUS' column to integer type")
    
    # Display columns after encoding
    print("\nColumns after encoding and dropping:")
    print(data.columns.tolist())
    
    return data, category_mappings

def extract_datetime_features(data):
    """
    Extract features from date and time columns.
    """
    # Convert 'OCC_DATE' to datetime
    data['OCC_DATE'] = pd.to_datetime(data['OCC_DATE'], errors='coerce')
    data = data.dropna(subset=['OCC_DATE'])
    print(f"\nConverted 'OCC_DATE' to datetime and dropped rows with NaT")
    
    # Extract features from 'OCC_DATE'
    data['OCC_YEAR'] = data['OCC_DATE'].dt.year
    data['OCC_MONTH'] = data['OCC_DATE'].dt.month
    data['OCC_DAY'] = data['OCC_DATE'].dt.day
    data['OCC_DOW'] = data['OCC_DATE'].dt.dayofweek  
    data['OCC_DOY'] = data['OCC_DATE'].dt.dayofyear
    data['OCC_HOUR'] = data['OCC_DATE'].dt.hour
    print("Extracted datetime features: 'OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_DOW', 'OCC_DOY', 'OCC_HOUR'")
    
    # Handle 'REPORT_DATE' if present
    if 'REPORT_DATE' in data.columns:
        data['REPORT_DATE'] = pd.to_datetime(data['REPORT_DATE'], errors='coerce')
        data = data.dropna(subset=['REPORT_DATE'])
        data['REPORT_YEAR'] = data['REPORT_DATE'].dt.year
        data['REPORT_MONTH'] = data['REPORT_DATE'].dt.month
        data['REPORT_DAY'] = data['REPORT_DATE'].dt.day
        data['REPORT_DOW'] = data['REPORT_DATE'].dt.dayofweek
        data['REPORT_DOY'] = data['REPORT_DATE'].dt.dayofyear
        data['REPORT_HOUR'] = data['REPORT_DATE'].dt.hour
        print("Extracted datetime features from 'REPORT_DATE': 'REPORT_YEAR', 'REPORT_MONTH', 'REPORT_DAY', 'REPORT_DOW', 'REPORT_DOY', 'REPORT_HOUR'")
        data = data.drop(columns=['REPORT_DATE'])
        print("Dropped 'REPORT_DATE' after extracting features")
    

    data = data.drop(columns=['OCC_DATE'])
    print("Dropped 'OCC_DATE' after extracting features")
    
    return data

def scale_numerical_data(data):
    """
    Scale numerical columns using StandardScaler.
    """
    numerical_cols = ['OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_DOW',
                      'OCC_HOUR', 'BIKE_COST', 'LONG_WGS84', 'LAT_WGS84', 'x', 'y',
                      'OCC_DOY', 'REPORT_YEAR', 'REPORT_MONTH', 'REPORT_DOW',
                      'REPORT_DAY', 'REPORT_DOY', 'REPORT_HOUR']
    
    # Filter numerical columns present in the data
    numerical_cols = [col for col in numerical_cols if col in data.columns]
    
    # Convert numerical columns to float explicitly to avoid dtype incompatibility
    data[numerical_cols] = data[numerical_cols].astype(float)
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Fit and transform the numerical columns
    scaled_values = scaler.fit_transform(data[numerical_cols])
    
    # Create a DataFrame for scaled values with appropriate column names and index
    scaled_df = pd.DataFrame(scaled_values, columns=numerical_cols, index=data.index)
    
    # Replace original numerical columns with scaled values
    data = data.drop(columns=numerical_cols).join(scaled_df)
    
    print(f"Scaled numerical columns: {numerical_cols}")
    
    # Verify data types after scaling
    print("\nData types after scaling:")
    print(data[numerical_cols].dtypes)
    
    return data

def transform_data(filename):
    """
    Complete data transformation pipeline.
    """
    # Load data
    data = load_data(filename)
    
    # Make a copy to avoid SettingWithCopyWarning
    data = data.copy()
    print("Made a copy of the data to avoid SettingWithCopyWarning")
    
    # Handle missing data
    data = handle_missing_data(data)
    
    # Extract datetime features
    data = extract_datetime_features(data)
    
    # Encode categorical data
    data, category_mappings = encode_categorical_data(data)
    
    # Scale numerical data
    data = scale_numerical_data(data)
    
    # Reset index
    data = data.reset_index(drop=True)
    print("Reset index of the DataFrame")
    
    # Save transformed data for future use
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    transformed_data_path = os.path.join(data_dir, 'transformed_theft.csv')
    data.to_csv(transformed_data_path, index=False)
    print(f"Transformed data saved to '{transformed_data_path}'")
    
    return data

if __name__ == "__main__":
    # For testing purposes
    try:
        transformed_data = transform_data('theft.csv')
        print("\nTransformed Data Sample:")
        print(transformed_data.head())
        
        # Check for missing values
        print("\nMissing values after transformation:")
        print(transformed_data.isnull().sum())
        
        # Check data types
        print("\nData types after transformation:")
        print(transformed_data.dtypes)
    except Exception as e:
        print(f"Error during data transformation: {e}")