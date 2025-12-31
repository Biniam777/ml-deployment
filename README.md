# Breast Cancer Prediction Project

This project is a Machine Learning deployment example that predicts breast cancer based on input features. It consists of a FastAPI backend and a simple HTML frontend.

## Project Structure

- **backend/**: Contains the FastAPI application and ML models.
  - `main.py`: The API server code.
  - `models/`: Directory containing pre-trained `.joblib` models (Logistic Regression, Decision Tree) and a scaler.
- **frontend/**: Contains the user interface.
  - `index.html`: A simple HTML page to interact with the API.

## Prerequisites

- Python 3.x
- pip

## Setup and Installation

1. **Navigate to the project folder.**

2. **Navigate to the backend directory:**

   ```bash
   cd backend
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the Backend Server:**
   From the `backend` directory, run:

   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at `http://127.0.0.1:8000`.

2. **Open the Frontend:**
   Open `frontend/index.html` in your web browser. You can simply double-click the file or serve it using a simple HTTP server.

## API Endpoints

- `GET /`: Health check. Returns a welcome message.
- `POST /predict/logistic`: Predicts using the Logistic Regression model.
- `POST /predict/tree`: Predicts using the Decision Tree model.

### Input Format

Both prediction endpoints expect a JSON body with a list of 5 float features:

```json
{
  "features": [
    14.0, // mean radius
    20.5, // mean texture
    90.0, // mean perimeter
    600.0, // mean area
    0.1 // mean smoothness
  ]
}
```

## Features Used

The models expect the following 5 features in this order:

1. Mean Radius
2. Mean Texture
3. Mean Perimeter
4. Mean Area
5. Mean Smoothness
