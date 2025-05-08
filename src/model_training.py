import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, roc_curve, roc_auc_score, classification_report)
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import from local modules
from src.visualization import plot_feature_importance, plot_model_performance_comparison

def create_model_directory():
    """Create models directory if it doesn't exist"""
    Path('models').mkdir(parents=True, exist_ok=True)
    print("Models directory created/verified.")

def create_results_directory():
    """Create results directory if it doesn't exist"""
    Path('results').mkdir(parents=True, exist_ok=True)
    print("Results directory created/verified.")

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and test sets and scale features
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple (np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler)
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data split into training ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples) sets")
    print(f"Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def get_models():
    """
    Get a dictionary of models to train
    
    Returns:
    --------
    dict
        Dictionary of model instances
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    return models

def evaluate_model(model, X_train, X_test, y_train, y_test, cv=5):
    """
    Evaluate a single model
    
    Parameters:
    -----------
    model : sklearn model
        Model to evaluate
    X_train : np.ndarray
        Scaled training features
    X_test : np.ndarray
        Scaled test features
    y_train : np.ndarray or pd.Series
        Training target values
    y_test : np.ndarray or pd.Series
        Test target values
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Collect metrics
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc,
        'CV Mean': cv_mean,
        'CV Std': cv_std
    }
    
    return metrics

def plot_confusion_matrix(model, X_test, y_test, model_name):
    """
    Plot confusion matrix for a model
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : np.ndarray
        Scaled test features
    y_test : np.ndarray or pd.Series
        Test target values
    model_name : str
        Name of the model
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Not Approved', 'Approved'],
        yticklabels=['Not Approved', 'Approved']
    )
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Create plots directory if it doesn't exist
    Path('plots').mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(f'plots/confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.close()

def plot_roc_curve(model, X_test, y_test, model_name):
    """
    Plot ROC curve for a model
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : np.ndarray
        Scaled test features
    y_test : np.ndarray or pd.Series
        Test target values
    model_name : str
        Name of the model
    """
    # Make predictions (probabilities)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14)
    plt.legend(loc="lower right")
    
    # Create plots directory if it doesn't exist
    Path('plots').mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(f'plots/roc_curve_{model_name.replace(" ", "_").lower()}.png')
    plt.close()

def train_evaluate_models(X, y):
    """
    Train and evaluate multiple models
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
        
    Returns:
    --------
    tuple (str, dict, pd.DataFrame)
        best_model_name, model_data, results_df
    """
    print("\nTraining and evaluating models...")
    
    # Create directory for models
    create_model_directory()
    
    # Create directory for results
    create_results_directory()
    
    # Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Get models
    models = get_models()
    
    # Initialize results dictionary
    results = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'AUC': [],
        'CV Mean': [],
        'CV Std': []
    }
    
    # Initialize dictionary to store trained models
    trained_models = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining and evaluating {name}...")
        
        # Evaluate model
        metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Store results
        results['Model'].append(name)
        for metric, value in metrics.items():
            results[metric].append(value)
        
        # Store trained model
        trained_models[name] = model
        
        # Print results
        print(f"Performance metrics for {name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Print classification report
        y_pred = model.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plot_confusion_matrix(model, X_test_scaled, y_test, name)
        
        # Plot ROC curve
        plot_roc_curve(model, X_test_scaled, y_test, name)
        
        # Plot feature importance for tree-based models
        if name in ['Random Forest', 'Gradient Boosting']:
            plot_feature_importance(model, X.columns, name)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv('results/model_comparison_results.csv', index=False)
    print("\nModel comparison results saved to 'results/model_comparison_results.csv'")
    
    # Create performance comparison chart
    plot_model_performance_comparison(results_df)
    
    # Find best model based on F1 score
    best_model_idx = results_df['F1 Score'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    best_model = trained_models[best_model_name]
    
    print(f"\nBest performing model: {best_model_name}")
    print(f"F1 Score: {results_df.loc[best_model_idx, 'F1 Score']:.4f}")
    
    # Save best model with necessary components
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'model_name': best_model_name
    }
    
    joblib.dump(model_data, 'models/best_loan_prediction_model.pkl')
    print(f"Best model ({best_model_name}) saved to 'models/best_loan_prediction_model.pkl'")
    
    return best_model_name, model_data, results_df

def load_best_model():
    """
    Load the best trained model
    
    Returns:
    --------
    dict or None
        Dictionary containing model components or None if model file doesn't exist
    """
    model_path = 'models/best_loan_prediction_model.pkl'
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model_data = joblib.load(model_path)
        print(f"Loaded {model_data['model_name']} model")
        return model_data
    else:
        print(f"Model file not found: {model_path}")
        return None

if __name__ == "__main__":
    # For testing purposes
    import os
    from data_processing import preprocess_data
    from feature_engineering import engineer_features, prepare_features_target
    
    file_path = "data/train_u6lujuX_CVtuZ9i.csv"
    
    if os.path.exists(file_path):
        df_preprocessed = preprocess_data(file_path)
        df_engineered = engineer_features(df_preprocessed)
        X, y = prepare_features_target(df_engineered)
        best_model_name, model_data, results_df = train_evaluate_models(X, y)
        print("\nModel training test completed successfully.")
    else:
        print(f"File not found: {file_path}")