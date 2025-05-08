import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set style for plots
plt.style.use('ggplot')
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']

def create_visualization_directory():
    """Create the plots directory if it doesn't exist"""
    Path('plots').mkdir(parents=True, exist_ok=True)
    print("Visualization directory created/verified.")

def plot_loan_status_distribution(df):
    """
    Plot the distribution of the target variable (Loan_Status)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with Loan_Status column
    """
    if 'Loan_Status' not in df.columns:
        print("Error: Loan_Status column not found in DataFrame")
        return
    
    plt.figure(figsize=(10, 6))
    loan_status_counts = df['Loan_Status'].value_counts()
    
    ax = sns.barplot(
        x=loan_status_counts.index.astype(str), 
        y=loan_status_counts.values, 
        palette=['#ff9999', '#66b3ff']
    )
    
    plt.title('Loan Status Distribution', fontsize=15)
    plt.xlabel('Loan Status (0=Not Approved, 1=Approved)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add labels on top of bars
    for i, v in enumerate(loan_status_counts.values):
        plt.text(i, v+5, str(v), ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('plots/loan_status_distribution.png')
    plt.close()
    
    print("Loan status distribution plot saved to 'plots/loan_status_distribution.png'")

def plot_categorical_features(df):
    """
    Plot categorical features against Loan_Status
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with categorical features and Loan_Status
    """
    if 'Loan_Status' not in df.columns:
        print("Error: Loan_Status column not found in DataFrame")
        return
    
    # Define categorical features
    categorical_features = [
        'Gender', 'Married', 'Education', 'Self_Employed', 
        'Property_Area', 'Has_Credit_History'
    ]
    
    # Filter only available features
    available_features = [f for f in categorical_features if f in df.columns]
    
    # Calculate number of rows and columns for subplots
    n_features = len(available_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(available_features):
        # Create a crosstab between the feature and Loan_Status
        crosstab = pd.crosstab(df[feature], df['Loan_Status'])
        crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
        
        ax = axes[i]
        crosstab_pct.plot(kind='bar', stacked=True, ax=ax, color=['#ff9999', '#66b3ff'])
        
        ax.set_title(f'{feature} vs Loan Status', fontsize=14)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.legend(['Not Approved', 'Approved'])
        
        # Add percentage labels
        for j, p in enumerate(ax.patches):
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy() 
            if height > 5:  # Only show labels for bars with height > 5%
                ax.text(x + width/2, 
                        y + height/2, 
                        f'{height:.1f}%', 
                        ha='center', 
                        va='center')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('plots/categorical_features.png')
    plt.close()
    
    print("Categorical features plot saved to 'plots/categorical_features.png'")

def plot_numerical_features(df):
    """
    Plot distribution of numerical features by Loan_Status
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with numerical features and Loan_Status
    """
    if 'Loan_Status' not in df.columns:
        print("Error: Loan_Status column not found in DataFrame")
        return
    
    # Define numerical features
    numerical_features = [
        'ApplicantIncome', 'LoanAmount', 'TotalIncome', 
        'Income_Loan_Ratio', 'EMI', 'Balance_Income'
    ]
    
    # Filter only available features
    available_features = [f for f in numerical_features if f in df.columns]
    
    # Calculate number of rows and columns for subplots
    n_features = len(available_features)
    n_cols = min(2, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(available_features):
        ax = axes[i]
        
        # Plot KDE for each loan status
        sns.kdeplot(
            df[df['Loan_Status'] == 0][feature], 
            ax=ax, 
            color='#ff9999', 
            label='Not Approved'
        )
        sns.kdeplot(
            df[df['Loan_Status'] == 1][feature], 
            ax=ax, 
            color='#66b3ff', 
            label='Approved'
        )
        
        ax.set_title(f'Distribution of {feature} by Loan Status', fontsize=14)
        ax.set_xlabel(feature, fontsize=12)
        ax.legend()
        
        # Add vertical lines for medians
        median_0 = df[df['Loan_Status'] == 0][feature].median()
        median_1 = df[df['Loan_Status'] == 1][feature].median()
        
        ax.axvline(x=median_0, color='#ff9999', linestyle='--', alpha=0.7)
        ax.axvline(x=median_1, color='#66b3ff', linestyle='--', alpha=0.7)
        
        # Add median annotations
        ax.text(
            median_0, 
            ax.get_ylim()[1] * 0.9, 
            f'Median: {median_0:.2f}', 
            color='#ff9999', 
            ha='right'
        )
        ax.text(
            median_1, 
            ax.get_ylim()[1] * 0.8, 
            f'Median: {median_1:.2f}', 
            color='#66b3ff', 
            ha='left'
        )
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('plots/numerical_features.png')
    plt.close()
    
    print("Numerical features plot saved to 'plots/numerical_features.png'")

def plot_correlation_matrix(df):
    """
    Plot correlation matrix for all features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with features
    """
    # Drop non-numeric columns
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])
    
    # Calculate correlation matrix
    correlation_matrix = df.corr()
    
    # Create mask for upper triangle
    mask = np.zeros_like(correlation_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        mask=mask, 
        cmap='coolwarm', 
        fmt='.2f',
        linewidths=0.5, 
        vmin=-1, 
        vmax=1
    )
    
    plt.title('Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    print("Correlation matrix plot saved to 'plots/correlation_matrix.png'")

def plot_feature_importance(model, feature_names, model_name):
    """
    Plot feature importance for tree-based models
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    model_name : str
        Name of the model
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print(f"Error: {model_name} does not have feature_importances_ attribute")
        return
    
    # Create DataFrame with feature names and importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot top 15 features (or all if less than 15)
    top_n = min(15, len(feature_importance))
    top_features = feature_importance.head(top_n)
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x='Importance', 
        y='Feature', 
        data=top_features,
        palette='viridis'
    )
    
    plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=15)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    
    # Add importance values
    for i, v in enumerate(top_features['Importance']):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig(f'plots/feature_importance_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    print(f"Feature importance plot for {model_name} saved to 'plots/feature_importance_{model_name.replace(' ', '_').lower()}.png'")
    
    return feature_importance

def plot_model_performance_comparison(results_df):
    """
    Plot performance comparison for multiple models
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with model performance results
    """
    # Check if DataFrame has required columns
    required_columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    if not all(col in results_df.columns for col in required_columns):
        print("Error: results_df is missing required columns")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Get metrics (excluding 'Model')
    metrics = [col for col in required_columns if col != 'Model']
    
    # Set up bar positions
    x = np.arange(len(results_df['Model']))
    width = 0.15
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        plt.bar(
            x + i*width, 
            results_df[metric], 
            width, 
            label=metric, 
            color=colors[i % len(colors)]
        )
    
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=15)
    plt.xticks(x + width*2, results_df['Model'], rotation=45)
    plt.legend(loc='best')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plots/model_performance_comparison.png')
    plt.close()
    
    print("Model performance comparison plot saved to 'plots/model_performance_comparison.png'")

def visualize_data(df):
    """
    Create all data visualizations
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with features and target
    """
    print("\nCreating data visualizations...")
    
    # Create visualization directory
    create_visualization_directory()
    
    # Create all plots
    plot_loan_status_distribution(df)
    plot_categorical_features(df)
    plot_numerical_features(df)
    plot_correlation_matrix(df)
    
    print("Data visualization completed successfully.")

if __name__ == "__main__":
    # For testing purposes
    import os
    from data_processing import preprocess_data
    from feature_engineering import engineer_features
    
    file_path = "data/train_u6lujuX_CVtuZ9i.csv"
    
    if os.path.exists(file_path):
        df_preprocessed = preprocess_data(file_path)
        df_engineered = engineer_features(df_preprocessed)
        visualize_data(df_engineered)
        print("\nVisualization test completed successfully.")
    else:
        print(f"File not found: {file_path}")