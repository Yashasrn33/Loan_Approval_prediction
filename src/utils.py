import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorama
from colorama import Fore, Style, init

# Initialize colorama
init()

def setup_environment():
    """Set up the environment for the application"""
    # Create necessary directories
    for directory in ['data', 'models', 'plots', 'results']:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Set matplotlib style
    plt.style.use('ggplot')
    
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.precision', 4)
    
    print("Environment setup complete.")

def print_header(title, width=60):
    """Print a formatted header"""
    print("\n" + "="*width)
    padding = (width - len(title)) // 2
    print(" "*padding + title)
    print("="*width + "\n")

def print_colored(text, color=Fore.WHITE, bold=False):
    """Print colored text using colorama"""
    if bold:
        print(f"{color}{Style.BRIGHT}{text}{Style.RESET_ALL}")
    else:
        print(f"{color}{text}{Style.RESET_ALL}")

def print_step(step_name, step_number=None, total_steps=None):
    """Print a step in the process"""
    if step_number is not None and total_steps is not None:
        prefix = f"[{step_number}/{total_steps}]"
    else:
        prefix = ">>>"
    
    print_colored(f"\n{prefix} {step_name}", Fore.CYAN, bold=True)

def print_success(message):
    """Print a success message"""
    print_colored(f"✅ {message}", Fore.GREEN)

def print_error(message):
    """Print an error message"""
    print_colored(f"❌ {message}", Fore.RED, bold=True)

def print_warning(message):
    """Print a warning message"""
    print_colored(f"⚠️ {message}", Fore.YELLOW)

def print_info(message):
    """Print an information message"""
    print_colored(f"ℹ️ {message}", Fore.BLUE)

def format_currency(amount):
    """Format amount as currency"""
    return f"${amount:,.2f}"

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.2f}%"

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def save_dataframe(df, filename, directory='results'):
    """
    Save DataFrame to CSV file
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filename : str
        Filename (without path)
    directory : str, default='results'
        Directory to save file in
    """
    # Create directory if it doesn't exist
    Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Full path
    file_path = os.path.join(directory, filename)
    
    # Save DataFrame
    df.to_csv(file_path, index=False)
    
    print_success(f"DataFrame saved to {file_path}")

def load_dataframe(filename, directory='results'):
    """
    Load DataFrame from CSV file
    
    Parameters:
    -----------
    filename : str
        Filename (without path)
    directory : str, default='results'
        Directory to load file from
        
    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame or None if file doesn't exist
    """
    file_path = os.path.join(directory, filename)
    
    if not os.path.exists(file_path):
        print_error(f"File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    print_success(f"DataFrame loaded from {file_path}")
    
    return df

def get_user_confirmation(message="Do you want to continue?"):
    """
    Get confirmation from user
    
    Parameters:
    -----------
    message : str, default="Do you want to continue?"
        Message to display
        
    Returns:
    --------
    bool
        True if user confirms, False otherwise
    """
    valid_responses = {
        'yes': True, 'y': True, 'ye': True,
        'no': False, 'n': False
    }
    
    while True:
        print_colored(f"\n{message} (yes/no): ", Fore.YELLOW)
        choice = input().lower().strip()
        
        if choice in valid_responses:
            return valid_responses[choice]
        
        print_warning("Please respond with 'yes' or 'no'")

if __name__ == "__main__":
    # Test the utilities
    setup_environment()
    
    print_header("LOAN PREDICTION SYSTEM")
    
    print_step("Data loading", 1, 5)
    print_success("Data loaded successfully")
    
    print_step("Data preprocessing", 2, 5)
    print_warning("Missing values detected in some columns")
    
    print_step("Feature engineering", 3, 5)
    print_info("Created 10 new features")
    
    print_step("Model training", 4, 5)
    print_error("Error occurred during model training")
    
    print_step("Prediction", 5, 5)
    print_success("Prediction completed successfully")
    
    print("\nFormatting examples:")
    print(f"Currency: {format_currency(12345.67)}")
    print(f"Percentage: {format_percentage(75.5)}")
    
    confirmed = get_user_confirmation("Did you like these utilities?")
    if confirmed:
        print_success("Great! Glad you liked them.")
    else:
        print_info("Thanks for your feedback. We'll work on improving them.")