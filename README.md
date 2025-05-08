# Loan Approval Prediction System

A comprehensive machine learning system for predicting loan approval status.

## Overview

This project implements a machine learning pipeline for predicting whether a loan application will be approved or rejected. The system includes data preprocessing, feature engineering, model training, evaluation, and deployment as a web application.

## Features

- **Data Preprocessing**: Handles missing values, encodes categorical variables
- **Feature Engineering**: Creates derived features like income-to-loan ratio and EMI calculations
- **Model Training**: Trains multiple classification models
- **Model Comparison**: Evaluates models using accuracy, precision, recall, F1 score, and AUC
- **Visualization**: Creates plots for data insights and model performance
- **REST API**: Provides an API endpoint for making predictions
- **Web Interface**: User-friendly interface for making predictions
- **Docker Deployment**: Easy deployment with Docker and Docker Compose

## File Structure

```
loan_prediction_system/
│
├── data/                          # Data directory
│   └── dataset.csv                # Dataset file
│
├── models/                        # Directory for saved models
│   └── best_loan_prediction_model.pkl  # Trained model file
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
├── templates/                     # HTML templates for web interface
│   └── index.html                 # Main web interface
│
├── app.py                         # Flask application for deployment
├── main.py                        # Main script to run the training pipeline
├── predict_cli.py                 # CLI for making predictions
├── Dockerfile                     # Docker container configuration
├── docker-compose.yml             # Docker Compose configuration
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Installation for Development

1. Clone the repository:

   ```
   git clone https://github.com/Yashasrn33/Loan_Approval_prediction.git
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

## Usage for Development

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

### Making Predictions via CLI

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

## Deployment

### Prerequisites for Deployment

Before deploying, ensure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)
- A trained model file (`best_loan_prediction_model.pkl`)

### Deployment Steps

1. **Prepare the Model**

Before deploying, ensure you have a trained model:

```bash
# Run the training pipeline if you haven't already
python main.py
```

This will create the `models/best_loan_prediction_model.pkl` file needed for deployment.

2. **Build and Run with Docker Compose**

```bash
# Build and start the container
docker-compose up --build -d

# View logs (optional)
docker-compose logs -f
```

3. **Access the Application**

Once deployed, the application will be available at:

- Web interface: http://localhost:5001
- API endpoint: http://localhost:5001/api/predict

### API Usage

The API accepts POST requests to the `/api/predict` endpoint with JSON data:

```json
{
  "Gender": 1,
  "Married": 1,
  "Dependents": 2,
  "Education": 1,
  "Self_Employed": 0,
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 1500,
  "LoanAmount": 120,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": 2
}
```

Example using curl:

```bash
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Gender": 1, "Married": 1, "Dependents": 2, "Education": 1, "Self_Employed": 0, "ApplicantIncome": 5000, "CoapplicantIncome": 1500, "LoanAmount": 120, "Loan_Amount_Term": 360, "Credit_History": 1, "Property_Area": 2}'
```

Example response:

```json
{
  "prediction": "Approved",
  "probability": 0.85,
  "features": {
    "credit_history": "Good",
    "income_loan_ratio": 54.17,
    "total_income": 6500.0,
    "emi": 333.33,
    "balance_income": 6166.67,
    "dti_ratio": 5.13
  },
  "explanation": [
    "Good credit history significantly increases approval chances",
    "Very high income-to-loan ratio indicates excellent repayment capacity",
    "Positive income balance after EMI payment",
    "Healthy debt-to-income ratio below 30%"
  ]
}
```

## Cloud Deployment Options

### AWS Elastic Beanstalk

1. Install the EB CLI: `pip install awsebcli`
2. Initialize EB: `eb init -p docker loan-prediction`
3. Create and deploy: `eb create loan-prediction-env`

### Google Cloud Run

1. Build and tag the image: `docker build -t gcr.io/[PROJECT_ID]/loan-prediction .`
2. Push to Container Registry: `docker push gcr.io/[PROJECT_ID]/loan-prediction`
3. Deploy to Cloud Run: `gcloud run deploy loan-prediction --image gcr.io/[PROJECT_ID]/loan-prediction --platform managed`

### Azure Container Instances

1. Create a container registry: `az acr create --name loanprediction --resource-group myResourceGroup --sku Basic`
2. Build and push: `az acr build --registry loanprediction --image loan-prediction:latest .`
3. Deploy: `az container create --resource-group myResourceGroup --name loan-prediction-container --image loanprediction.azurecr.io/loan-prediction:latest --dns-name-label loan-prediction --ports 5000`

## Monitoring and Maintenance

- **Logs**: View logs with `docker-compose logs`
- **Updates**: To update the application, rebuild and restart the containers with `docker-compose up --build -d`
- **Backup**: Regularly backup your model file to prevent data loss

## Troubleshooting

- **Container fails to start**: Check logs with `docker-compose logs`
- **Model loading error**: Ensure the model file path is correct and the file exists
- **API errors**: Verify the JSON format matches the expected input structure
- **NumPy version issues**: Make sure the NumPy version in requirements.txt matches the version used to train the model

## Security Considerations

For production deployments:

1. Use HTTPS with a valid SSL certificate
2. Implement proper authentication for API access
3. Consider adding rate limiting to prevent abuse
4. Remove debug mode in Flask by setting `debug=False`
5. Regularly update dependencies to patch security vulnerabilities

<img width="1097" alt="Screenshot 2025-05-08 at 1 36 01 AM" src="https://github.com/user-attachments/assets/9176a34b-3f3f-4298-8f5d-3962c7b29742" />

<img width="1051" alt="Screenshot 2025-05-08 at 1 36 20 AM" src="https://github.com/user-attachments/assets/c44947e2-7f96-40f2-a9d7-e5b6c5aac7a7" />

<img width="963" alt="Screenshot 2025-05-08 at 1 36 33 AM" src="https://github.com/user-attachments/assets/a43eb4c0-f5a3-4670-b763-44d3cf87cc1a" />

## Future Enhancements

1. Add user authentication for API access
2. Implement feature importance visualization in the web interface
3. Add A/B testing capability for model comparison
4. Implement model monitoring for drift detection
5. Create a model retraining pipeline
