# Loan Prediction System

A comprehensive machine learning system for predicting loan approval status.

## Overview

This project implements a machine learning pipeline for predicting whether a loan application will be approved or rejected. The system includes data preprocessing, feature engineering, model training, and evaluation. Multiple models are trained and compared to find the best performer.

## Features

- **Data Preprocessing**: Handles missing values, encodes categorical variables
- **Feature Engineering**: Creates derived features like income-to-loan ratio and EMI calculations
- **Model Training**: Trains multiple classification models
- **Model Comparison**: Evaluates models using accuracy, precision, recall, F1 score, and AUC
- **Visualization**: Creates plots for data insights and model performance
- **Prediction Interface**: Provides a CLI for making predictions with new data

## File Structure

```
loan_prediction_system/
│
├── data/                          # Data directory
│   └── dataset.csv                # Dataset file
│
├── models/                        # Directory for saved models
│   └── .gitkeep                   # To ensure directory is tracked in git
│
├── plots/                         # Directory for visualizations
│   └── .gitkeep                   # To ensure directory is tracked in git
│
├── results/                       # Directory for results
│   └── .gitkeep                   # To ensure directory is tracked in git
│
├── src/                           # Source code
│   ├── __init__.py                # Makes src a Python package
│   ├── data_processing.py         # Data loading and preprocessing functions
│   ├── visualization.py           # Data visualization functions
│   ├── feature_engineering.py     # Feature engineering functions
│   ├── model_training.py          # Model training and evaluation
│   ├── prediction.py              # Functions for making predictions
│   └── utils.py                   # Utility functions
│
├── main.py                        # Main script to run the application
├── predict_cli.py                 # CLI for making predictions
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/loan-prediction-system.git
   cd loan-prediction-system
   ```

2. Create a virtual environment and install dependencies:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Place your dataset in the data directory:
   ```
   cp path/to/your/dataset.csv data/
   ```

## Usage

### Training Models

To train models and select the best one:

```
python main.py
```

This will:

1. Load and preprocess the data
2. Create visualizations in the 'plots' directory
3. Train multiple models
4. Compare models and select the best one
5. Save the best model to the 'models' directory

### Making Predictions

To make predictions using the trained model:

```
python predict_cli.py
```

Follow the prompts to enter applicant details and get a prediction.

## Models

The system trains and compares the following models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Naive Bayes

## Results

Model performance results are saved in:

- CSV format in 'results/model_comparison_results.csv'
- Visualizations in the 'plots' directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.
