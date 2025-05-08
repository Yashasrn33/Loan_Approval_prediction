import pandas as pd
import numpy as np

def create_basic_features(df):
    """
    Create basic derived features from the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added features
    """
    # Make a copy to avoid modifying the original
    df_new = df.copy()
    
    # Create total income feature
    df_new['TotalIncome'] = df_new['ApplicantIncome'] + df_new['CoapplicantIncome']
    
    # Create income to loan ratio (important for loan approval)
    df_new['Income_Loan_Ratio'] = df_new['TotalIncome'] / df_new['LoanAmount'].apply(lambda x: max(x, 1))
    
    # Create binary feature for credit history (most important feature)
    df_new['Has_Credit_History'] = df_new['Credit_History'].apply(lambda x: 1 if x >= 1 else 0)
    
    print("Created basic features:")
    print("- TotalIncome: Sum of applicant and coapplicant income")
    print("- Income_Loan_Ratio: Ratio of total income to loan amount")
    print("- Has_Credit_History: Binary indicator of credit history")
    
    return df_new

def create_financial_features(df):
    """
    Create advanced financial features for loan prediction
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added financial features
    """
    # Make a copy to avoid modifying the original
    df_new = df.copy()
    
    # Calculate EMI (Equated Monthly Installment)
    # Formula: [P x R x (1+R)^N]/[(1+R)^N-1], simplified for this case
    df_new['EMI'] = df_new['LoanAmount'] * 1000 / df_new['Loan_Amount_Term'].apply(lambda x: max(x, 1)) / 12
    
    # Calculate income remaining after EMI payment
    df_new['Balance_Income'] = df_new['TotalIncome'] - df_new['EMI']
    
    # Calculate debt-to-income ratio (EMI as % of income)
    df_new['DTI_Ratio'] = (df_new['EMI'] / df_new['TotalIncome'].apply(lambda x: max(x, 1))) * 100
    
    # Create income stability indicator (if coapplicant contributes significantly)
    df_new['Income_Stability'] = (df_new['CoapplicantIncome'] / df_new['TotalIncome'].apply(lambda x: max(x, 1))).apply(lambda x: min(x, 1))
    
    print("Created financial features:")
    print("- EMI: Estimated monthly installment")
    print("- Balance_Income: Income remaining after EMI payment")
    print("- DTI_Ratio: Debt-to-income ratio")
    print("- Income_Stability: Indicator of income stability based on coapplicant contribution")
    
    return df_new

def create_interaction_features(df):
    """
    Create interaction features between existing variables
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added interaction features
    """
    # Make a copy to avoid modifying the original
    df_new = df.copy()
    
    # Interaction between education and income
    df_new['Education_Income'] = df_new['Education'] * df_new['TotalIncome']
    
    # Interaction between credit history and loan amount
    df_new['Credit_Loan'] = df_new['Credit_History'] * df_new['LoanAmount']
    
    # Interaction between property area and loan amount
    df_new['Property_Loan'] = df_new['Property_Area'] * df_new['LoanAmount']
    
    # Interaction between married status and dependents
    if 'Dependents' in df_new.columns:
        df_new['Family_Size'] = df_new['Married'] + df_new['Dependents'].apply(lambda x: float(x))
    
    print("Created interaction features:")
    print("- Education_Income: Interaction between education and income")
    print("- Credit_Loan: Interaction between credit history and loan amount")
    print("- Property_Loan: Interaction between property area and loan amount")
    if 'Dependents' in df_new.columns:
        print("- Family_Size: Derived from married status and dependents")
    
    return df_new

def engineer_features(df):
    """
    Complete feature engineering pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input preprocessed DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all engineered features
    """
    print("\nStarting feature engineering...")
    
    # Apply feature engineering steps sequentially
    df_basic = create_basic_features(df)
    df_financial = create_financial_features(df_basic)
    df_interaction = create_interaction_features(df_financial)
    
    # Log the new features
    original_cols = set(df.columns)
    new_cols = set(df_interaction.columns) - original_cols
    print(f"\nAdded {len(new_cols)} new features:")
    print(", ".join(new_cols))
    
    print("\nFeature engineering completed successfully.")
    
    return df_interaction

def prepare_features_target(df):
    """
    Separate features and target from the DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with features and target
        
    Returns:
    --------
    tuple (pd.DataFrame, pd.Series)
        X: Features DataFrame
        y: Target Series
    """
    # Check if Loan_Status exists in the DataFrame
    if 'Loan_Status' not in df.columns:
        raise ValueError("Target variable 'Loan_Status' not found in DataFrame")
    
    # Remove non-feature columns
    drop_cols = ['Loan_ID', 'Loan_Status']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    # Create feature matrix and target vector
    X = df[feature_cols]
    y = df['Loan_Status']
    
    print(f"\nFeatures and target prepared:")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target distribution:\n{y.value_counts(normalize=True).apply(lambda x: f'{x:.2%}')}")
    
    return X, y

if __name__ == "__main__":
    # For testing purposes
    import os
    from data_processing import preprocess_data
    
    file_path = "data/train_u6lujuX_CVtuZ9i.csv"
    
    if os.path.exists(file_path):
        df_preprocessed = preprocess_data(file_path)
        df_engineered = engineer_features(df_preprocessed)
        X, y = prepare_features_target(df_engineered)
        print("\nFeature engineering test completed successfully.")
    else:
        print(f"File not found: {file_path}")