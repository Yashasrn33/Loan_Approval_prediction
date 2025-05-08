import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from pathlib import Path
import pickle
import sys

# Create Flask app
app = Flask(__name__)

# Load model
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/best_loan_prediction_model.pkl')

# Function to create a simple model for testing if loading the real model fails
def create_dummy_model():
    print("Creating a dummy model for testing...")
    
    class DummyModel:
        def predict(self, X):
            # Always predict 1 (approved)
            return np.ones(len(X))
        
        def predict_proba(self, X):
            # Return dummy probabilities (90% approved)
            return np.array([[0.1, 0.9]] * len(X))
    
    dummy_model = DummyModel()
    
    # Create dummy scaler that doesn't transform
    class DummyScaler:
        def transform(self, X):
            return X
    
    dummy_scaler = DummyScaler()
    
    # Create a sample feature list
    dummy_features = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
        'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'TotalIncome', 'Income_Loan_Ratio', 'EMI', 'Balance_Income'
    ]
    
    # Return a dummy model data dictionary
    return {
        'model': dummy_model,
        'scaler': dummy_scaler,
        'feature_names': dummy_features,
        'model_name': 'Dummy Model (Real model failed to load)'
    }

# Try to load the real model, fall back to dummy model if it fails
try:
    print(f"Attempting to load model from {MODEL_PATH}")
    
    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file not found: {MODEL_PATH}")
        model_data = create_dummy_model()
    else:
        # Try to load the model
        model_data = joblib.load(MODEL_PATH)
        print(f"Successfully loaded {model_data['model_name']} model")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Falling back to dummy model for testing")
    model_data = create_dummy_model()

# Extract model components
model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']
model_name = model_data['model_name']

print(f"Model ready: {model_name}")
print(f"Feature names: {feature_names}")

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
    try:
        # Prepare input data
        input_df = prepare_input_data(user_input)
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(input_df.columns)
        for feature in missing_features:
            input_df[feature] = 0
        
        # Select only the features used during training
        input_features = []
        for feature in feature_names:
            if feature in input_df.columns:
                input_features.append(feature)
            else:
                print(f"Warning: Feature '{feature}' not found in input data")
        
        input_df = input_df[input_features]
        
        # Scale features
        try:
            input_scaled = scaler.transform(input_df)
        except Exception as e:
            print(f"Error scaling features: {str(e)}")
            print("Using unscaled features")
            input_scaled = input_df
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Extract key factors for interpretation
        result = {
            'prediction': 'Approved' if prediction == 1 else 'Not Approved',
            'probability': float(probability),
            'features': {
                'credit_history': 'Good' if user_input['Credit_History'] == 1 else 'Poor',
                'income_loan_ratio': float(input_df['Income_Loan_Ratio'].values[0]),
                'total_income': float(input_df['TotalIncome'].values[0]),
                'emi': float(input_df['EMI'].values[0]),
                'balance_income': float(input_df['Balance_Income'].values[0]),
                'dti_ratio': float(input_df['DTI_Ratio'].values[0]) if 'DTI_Ratio' in input_df.columns else None
            }
        }
        
        # Add explanation based on key factors
        result['explanation'] = get_prediction_explanation(result)
        
        return result
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        # Return a fallback prediction with error message
        return {
            'error': True,
            'message': str(e),
            'prediction': 'Unknown',
            'probability': 0.5,
            'features': {
                'credit_history': 'Unknown',
                'income_loan_ratio': 0,
                'total_income': 0,
                'emi': 0,
                'balance_income': 0,
                'dti_ratio': 0
            },
            'explanation': ["An error occurred during prediction. Check server logs for details."]
        }

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

# API routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': model_name,
        'python_version': sys.version,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        input_data = request.json
        
        # Validate input data
        required_fields = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                          'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                          'Loan_Amount_Term', 'Credit_History', 'Property_Area']
        
        missing_fields = [field for field in required_fields if field not in input_data]
        
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields
            }), 400
        
        # Make prediction
        result = predict_loan_approval(input_data)
        
        # Check if prediction failed
        if 'error' in result and result['error']:
            return jsonify(result), 500
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in API: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)