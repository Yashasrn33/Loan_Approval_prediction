import pandas as pd
import numpy as np
import os
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'models', 'plots', 'results']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def load_data(file_path):
    """
    Load data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame
    """
    print(f"Loading data from {file_path}...")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    print(f"Dataset loaded with shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df

def check_missing_values(df):
    """
    Check for missing values in DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.Series
        Series containing count of missing values for each column
    """
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    })
    
    print("\nMissing values information:")
    print(missing_info[missing_info['Missing Values'] > 0])
    
    return missing_values

def handle_missing_values(df):
    """
    Handle missing values in DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with missing values
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with handled missing values
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Identify numerical and categorical columns
    numerical_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'ApplicantIncome', 'CoapplicantIncome']
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Education', 'Property_Area']
    
    # Fill missing values in numerical columns with median
    for col in numerical_cols:
        if col in df_processed.columns:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Fill missing values in categorical columns with mode
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    # Verify no missing values remain
    remaining_missing = df_processed.isnull().sum().sum()
    print(f"\nRemaining missing values after handling: {remaining_missing}")
    
    return df_processed

def encode_categorical_variables(df):
    """
    Encode categorical variables in DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with categorical variables
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with encoded categorical variables
    """
    # Make a copy to avoid modifying the original
    df_encoded = df.copy()
    
    # Convert target variable to numeric
    if 'Loan_Status' in df_encoded.columns:
        df_encoded.replace({'Loan_Status': {'N': 0, 'Y': 1}}, inplace=True)
    
    # Handle '3+' dependents
    if 'Dependents' in df_encoded.columns:
        df_encoded.replace(to_replace='3+', value=4, inplace=True)
    
    # Encode categorical variables
    encoding_mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
    }
    
    for col, mapping in encoding_mappings.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)
    
    return df_encoded

def preprocess_data(file_path):
    """
    Complete data preprocessing pipeline
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Fully preprocessed DataFrame
    """
    # Create necessary directories
    create_directories()
    
    # Load data
    df = load_data(file_path)
    
    # Check and handle missing values
    check_missing_values(df)
    df_processed = handle_missing_values(df)
    
    # Encode categorical variables
    df_encoded = encode_categorical_variables(df_processed)
    
    print("\nPreprocessing completed successfully.")
    
    return df_encoded

if __name__ == "__main__":
    # For testing purposes
    file_path = "data/train_u6lujuX_CVtuZ9i.csv"
    df_preprocessed = preprocess_data(file_path)
    print("\nPreprocessed data sample:")
    print(df_preprocessed.head())