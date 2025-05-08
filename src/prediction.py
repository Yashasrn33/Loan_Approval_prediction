import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

def load_model():
    """
    Load the trained model
    
    Returns:
    --------
    dict or None
        Dictionary containing model components or None if model file doesn't exist
    """
    model_path = 'models/best_loan_prediction_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    try:
        model_data = joblib.load(model_path)
        print(f"Model loaded: {model_data['model_name']}")
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def prepare_input_data(user_input):
    """
    Prepare user input data for prediction
    
    Parameters:
    -----------
    user_input : dict
        Dictionary containing user input values
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with processed input data
    """
    # Create a copy of the input to avoid modifying the original
    input_data = user_input.copy()
    
    # Create derived features
    # Total income
    input_data['TotalIncome'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
    
    # Income to loan ratio
    input_data['Income_Loan_Ratio'] = input_data['TotalIncome'] / max(input_data['LoanAmount'], 1)
    
    # EMI calculation
    input_data['EMI'] = input_data['LoanAmount'] * 1000 / max(input_data['Loan_Amount_Term'], 1) / 12
    
    # Balance income
    input_data['Balance_Income'] = input_data['TotalIncome'] - input_data['EMI']
    
    # Credit history indicator
    input_data['Has_Credit_History'] = 1 if input_data['Credit_History'] >= 1 else 0
    
    # Debt-to-income ratio
    input_data['DTI_Ratio'] = (input_data['EMI'] / max(input_data['TotalIncome'], 1)) * 100
    
    # Income stability
    input_data['Income_Stability'] = (input_data['CoapplicantIncome'] / max(input_data['TotalIncome'], 1))
    
    # Interaction features
    input_data['Education_Income'] = input_data['Education'] * input_data['TotalIncome']
    input_data['Credit_Loan'] = input_data['Credit_History'] * input_data['LoanAmount']
    input_data['Property_Loan'] = input_data['Property_Area'] * input_data['LoanAmount']
    
    # Family size (if Dependents is available)
    if 'Dependents' in input_data:
        input_data['Family_Size'] = input_data['Married'] + float(input_data['Dependents'])
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    return input_df

def predict_loan_approval(user_input):
    """
    Predict loan approval based on user input
    
    Parameters:
    -----------
    user_input : dict
        Dictionary containing user input values
        
    Returns:
    --------
    dict
        Prediction results with probability and key factors
    """
    # Load model
    model_data = load_model()
    
    if model_data is None:
        return {
            'error': True,
            'message': 'Failed to load model'
        }
    
    # Extract model components
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # Prepare input data
    input_df = prepare_input_data(user_input)
    
    # Ensure all required features are present
    missing_features = set(feature_names) - set(input_df.columns)
    for feature in missing_features:
        input_df[feature] = 0
    
    # Select only the features used during training
    input_df = input_df[feature_names]
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Extract key factors for interpretation
    result = {
        'prediction': 'Approved' if prediction == 1 else 'Not Approved',
        'probability': probability,
        'features': {
            'credit_history': 'Good' if user_input['Credit_History'] == 1 else 'Poor',
            'income_loan_ratio': input_df['Income_Loan_Ratio'].values[0],
            'total_income': input_df['TotalIncome'].values[0],
            'emi': input_df['EMI'].values[0],
            'balance_income': input_df['Balance_Income'].values[0],
            'dti_ratio': input_df['DTI_Ratio'].values[0] if 'DTI_Ratio' in input_df.columns else None
        }
    }
    
    # Add explanation based on key factors
    result['explanation'] = get_prediction_explanation(result)
    
    return result

def get_prediction_explanation(result):
    """
    Generate an explanation for the prediction
    
    Parameters:
    -----------
    result : dict
        Prediction result
        
    Returns:
    --------
    list
        List of explanation strings
    """
    explanation = []
    
    # Credit history explanation
    if result['features']['credit_history'] == 'Good':
        explanation.append("Good credit history significantly increases approval chances")
    else:
        explanation.append("Poor credit history significantly reduces approval chances")
    
    # Income to loan ratio explanation
    ratio = result['features']['income_loan_ratio']
    if ratio > 10:
        explanation.append("Very high income-to-loan ratio indicates excellent repayment capacity")
    elif ratio > 5:
        explanation.append("Good income-to-loan ratio indicates strong repayment capacity")
    elif ratio > 3:
        explanation.append("Moderate income-to-loan ratio indicates adequate repayment capacity")
    else:
        explanation.append("Low income-to-loan ratio indicates potential repayment risk")
    
    # Balance income explanation
    if result['features']['balance_income'] > 0:
        explanation.append("Positive income balance after EMI payment")
    else:
        explanation.append("Insufficient income to cover monthly EMI payment")
    
    # DTI ratio explanation
    if result['features']['dti_ratio'] is not None:
        dti = result['features']['dti_ratio']
        if dti < 30:
            explanation.append("Healthy debt-to-income ratio below 30%")
        elif dti < 40:
            explanation.append("Moderate debt-to-income ratio between 30-40%")
        else:
            explanation.append("High debt-to-income ratio above 40% may be concerning")
    
    return explanation

def format_currency(amount):
    """Format amount as currency"""
    return f"${amount:,.2f}"

def display_prediction_result(result):
    """
    Display prediction result in a user-friendly format
    
    Parameters:
    -----------
    result : dict
        Prediction result
    """
    if 'error' in result:
        print(f"\nError: {result['message']}")
        return
    
    print("\n" + "="*60)
    print(" "*20 + "PREDICTION RESULT")
    print("="*60 + "\n")
    
    # Display prediction and probability
    if result['prediction'] == 'Approved':
        print("LOAN STATUS: APPROVED ✅")
    else:
        print("LOAN STATUS: NOT APPROVED ❌")
    
    print(f"Approval Probability: {result['probability']*100:.2f}%\n")
    
    # Display key factors
    print("KEY FACTORS:")
    print(f"• Credit History: {result['features']['credit_history']}")
    print(f"• Income to Loan Ratio: {result['features']['income_loan_ratio']:.2f}")
    print(f"• Monthly EMI: {format_currency(result['features']['emi'])}")
    print(f"• Income Balance After EMI: {format_currency(result['features']['balance_income'])}")
    print(f"• Total Income: {format_currency(result['features']['total_income'])}")
    
    if result['features']['dti_ratio'] is not None:
        print(f"• Debt-to-Income Ratio: {result['features']['dti_ratio']:.2f}%")
    
    # Display explanation
    print("\nEXPLANATION:")
    for point in result['explanation']:
        print(f"• {point}")

if __name__ == "__main__":
    # For testing purposes
    test_input = {
        'Gender': 1,  # Male
        'Married': 1,  # Yes
        'Dependents': 2,
        'Education': 1,  # Graduate
        'Self_Employed': 0,  # No
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 1500,
        'LoanAmount': 120,  # in thousands
        'Loan_Amount_Term': 360,  # in months
        'Credit_History': 1,  # Has credit history
        'Property_Area': 2  # Urban
    }
    
    result = predict_loan_approval(test_input)
    display_prediction_result(result)