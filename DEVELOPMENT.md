# Loan Prediction System Deployment Guide

This guide explains how to deploy the Loan Prediction System as a Flask web application using Docker.

## Prerequisites

Before you begin, ensure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)
- A trained model file (`best_loan_prediction_model.pkl`)

## Project Structure

```
loan_prediction_system/
│
├── app.py                       # Flask application
├── Dockerfile                   # Docker container configuration
├── docker-compose.yml           # Docker Compose configuration
├── requirements.txt             # Python dependencies
├── models/                      # Directory for saved models
│   └── best_loan_prediction_model.pkl  # Trained model file
└── templates/                   # HTML templates
    └── index.html               # Web interface
```

## Deployment Steps

### 1. Prepare the Model

Before deploying, ensure you have a trained model:

```bash
# Run the training pipeline if you haven't already
python main.py
```

This will create the `models/best_loan_prediction_model.pkl` file needed for deployment.

### 2. Update Requirements File

Ensure your `requirements.txt` contains all necessary dependencies:

```
numpy==1.24.3
pandas==2.0.1
scikit-learn==1.2.2
flask==2.3.2
joblib==1.2.0
matplotlib==3.7.1
seaborn==0.12.2
```

### 3. Build and Run with Docker Compose

```bash
# Build and start the container
docker-compose up --build -d

# View logs (optional)
docker-compose logs -f
```

### 4. Access the Application

Once deployed, the application will be available at:

- Web interface: http://localhost:5001
- API endpoint: http://localhost:5001/api/predict

### 5. API Usage

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

## Security Considerations

For production deployments:

1. Use HTTPS with a valid SSL certificate
2. Implement proper authentication for API access
3. Consider adding rate limiting to prevent abuse
4. Remove debug mode in Flask by setting `debug=False`
5. Regularly update dependencies to patch security vulnerabilities
