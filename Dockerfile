# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files (if available)
COPY models/ ./models/

# Copy application code
COPY app.py .
COPY templates/ ./templates/

# Create necessary directories
RUN mkdir -p plots results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV MODEL_PATH=models/best_loan_prediction_model.pkl

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]