#!/usr/bin/env python
"""
Command Line Interface for Loan Prediction System
This script provides a user-friendly CLI for making loan approval predictions
"""

import os
import sys
from pathlib import Path
import joblib

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import local modules
from src.prediction import predict_loan_approval, display_prediction_result
from src.utils import (
    setup_environment, print_header, print_colored, print_step,
    print_success, print_error, print_warning, print_info,
    clear_screen, get_user_confirmation
)
from colorama import Fore, Style, init

# Initialize colorama
init()

def get_user_input():
    """Get loan applicant details from user"""
    print_header("LOAN APPROVAL PREDICTION")
    
    print_colored("Please enter the applicant details:\n", Fore.CYAN)
    
    try:
        # Get categorical inputs
        gender = input("Gender (Male/Female): ").strip()
        gender = 1 if gender.lower() == 'male' else 0
        
        married = input("Married (Yes/No): ").strip()
        married = 1 if married.lower() == 'yes' else 0
        
        dependents = input("Number of Dependents (0/1/2/3/3+): ").strip()
        dependents = 4 if dependents == '3+' else int(dependents)
        
        education = input("Education (Graduate/Not Graduate): ").strip()
        education = 1 if education.lower() == 'graduate' else 0
        
        self_employed = input("Self Employed (Yes/No): ").strip()
        self_employed = 1 if self_employed.lower() == 'yes' else 0
        
        property_area = input("Property Area (Rural/Semiurban/Urban): ").strip().lower()
        if property_area == 'rural':
            property_area = 0
        elif property_area == 'semiurban':
            property_area = 1
        else:
            property_area = 2
        
        # Get numerical inputs
        applicant_income = float(input("Applicant Income: ").strip())
        coapplicant_income = float(input("Coapplicant Income: ").strip())
        loan_amount = float(input("Loan Amount (in thousands): ").strip())
        loan_amount_term = float(input("Loan Amount Term (in months): ").strip())
        credit_history = int(input("Credit History meets guidelines (1=Yes, 0=No): ").strip())
        
        # Create user input dictionary
        user_input = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }
        
        return user_input
        
    except ValueError as e:
        print_error(f"Invalid input: {e}")
        print_warning("Please enter valid values for all fields.")
        return None

def main():
    """Main function for prediction CLI"""
    # Setup environment
    setup_environment()
    
    # Clear screen
    clear_screen()
    
    # Print welcome message
    print_header("LOAN PREDICTION SYSTEM")
    print_info("This program predicts loan approval based on applicant details.")
    
    # Check if model exists
    model_path = 'models/best_loan_prediction_model.pkl'
    if not os.path.exists(model_path):
        print_error(f"Model file not found: {model_path}")
        print_warning("Please run main.py first to train the model.")
        sys.exit(1)
    
    # Load model information
    try:
        model_data = joblib.load(model_path)
        print_success(f"Loaded {model_data['model_name']} model.")
    except Exception as e:
        print_error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Prediction loop
    while True:
        # Get user input
        user_input = get_user_input()
        
        if user_input is None:
            # Invalid input, ask if user wants to try again
            if get_user_confirmation("Do you want to try again?"):
                clear_screen()
                continue
            else:
                break
        
        # Make prediction
        result = predict_loan_approval(user_input)
        
        # Display result
        display_prediction_result(result)
        
        # Ask if user wants to make another prediction
        if not get_user_confirmation("Do you want to make another prediction?"):
            break
        
        clear_screen()
    
    print_header("THANK YOU FOR USING THE LOAN PREDICTION SYSTEM")

if __name__ == "__main__":
    main()