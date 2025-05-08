#!/usr/bin/env python
"""
Main script for Loan Prediction System

This script runs the entire pipeline from data loading to model training
"""

import os
import sys
from pathlib import Path
import time

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import local modules
from src.data_processing import preprocess_data
from src.feature_engineering import engineer_features, prepare_features_target
from src.visualization import visualize_data
from src.model_training import train_evaluate_models, load_best_model
from src.utils import (
    setup_environment, print_header, print_colored, print_step,
    print_success, print_error, print_warning, print_info,
    clear_screen, get_user_confirmation
)
from colorama import Fore, Style, init

# Initialize colorama
init()

def main():
    """Main function to run the loan prediction pipeline"""
    # Setup environment
    setup_environment()
    
    # Clear screen
    clear_screen()
    
    # Print welcome message
    print_header("LOAN PREDICTION SYSTEM")
    print_info("This program trains models to predict loan approval based on applicant details.")
    
    # Define file path
    file_path = os.path.join('data', 'dataset.csv')
    
    # Check if file exists
    if not os.path.exists(file_path):
        print_error(f"Dataset file not found: {file_path}")
        print_info("Please place your dataset file in the 'data' directory.")
        sys.exit(1)
    
    # Check if model already exists
    model_path = os.path.join('models', 'best_loan_prediction_model.pkl')
    if os.path.exists(model_path):
        print_warning(f"Model file already exists: {model_path}")
        
        if not get_user_confirmation("Do you want to retrain the model?"):
            print_info("Exiting without retraining.")
            sys.exit(0)
    
    # Start time
    start_time = time.time()
    
    # Step 1: Data Preprocessing
    print_step("Data Preprocessing", 1, 4)
    df_preprocessed = preprocess_data(file_path)
    print_success("Data preprocessing completed.")
    
    # Step 2: Feature Engineering
    print_step("Feature Engineering", 2, 4)
    df_engineered = engineer_features(df_preprocessed)
    X, y = prepare_features_target(df_engineered)
    print_success("Feature engineering completed.")
    
    # Step 3: Data Visualization
    print_step("Data Visualization", 3, 4)
    visualize_data(df_engineered)
    print_success("Data visualization completed.")
    
    # Step 4: Model Training and Evaluation
    print_step("Model Training and Evaluation", 4, 4)
    best_model_name, model_data, results_df = train_evaluate_models(X, y)
    print_success("Model training and evaluation completed.")
    
    # End time
    end_time = time.time()
    
    # Print summary
    print_header("PIPELINE SUMMARY")
    print_info(f"Total execution time: {(end_time - start_time):.2f} seconds")
    print_info(f"Best model: {best_model_name}")
    print_info(f"F1 Score: {results_df[results_df['Model'] == best_model_name]['F1 Score'].values[0]:.4f}")
    print_info(f"Visualizations saved in 'plots' directory")
    print_info(f"Results saved in 'results' directory")
    print_info(f"Model saved in 'models' directory")
    
    print_colored("\nYou can now use predict_cli.py to make predictions.", Fore.GREEN, True)
    print_colored("Example: python predict_cli.py", Fore.GREEN)

if __name__ == "__main__":
    main()