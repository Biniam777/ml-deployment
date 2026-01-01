# Use official Python image
FROM python:3.11-slim

# Set working directory inside container to backend
WORKDIR /app

# Copy only the backend folder contents into /app
COPY backend/ /app/

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn pandas scikit-learn joblib

# Expose port
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
