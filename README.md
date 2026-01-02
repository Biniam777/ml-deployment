# Breast Cancer Prediction System

A Machine Learning deployment project that predicts breast cancer diagnosis (Benign/Malignant) based on tumor measurements. The system features a FastAPI backend serving a scikit-learn model and an interactive HTML frontend.

## 🚀 Features

- **Dual Model Support**: Choose between Logistic Regression and Decision Tree models for prediction.
- **Interactive UI**: User-friendly web interface for easy data entry and visualization.
- **RESTful API**: robust backend API built with FastAPI.
- **Dockerized**: Ready for containerized deployment.

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **ML Libraries**: Scikit-learn, Pandas, NumPy, Joblib
- **Frontend**: HTML5, CSS3, JavaScript
- **Containerization**: Docker

## 📂 Project Structure

```
ml-deployment/
├── backend/
│   ├── models/                 # Pre-trained ML models (.joblib)
│   ├── static/                 # Frontend assets (index.html)
│   ├── Dockerfile              # Docker configuration
│   ├── main.py                 # FastAPI application entry point
│   ├── Procfile                # Deployment configuration
│   └── requirements.txt        # Python dependencies
└── README.md
```

## 🏁 Getting Started

### Prerequisites

- Python 3.11+
- pip
- Docker (optional)

### 🔧 Local Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ml-deployment
   ```

2. **Navigate to the backend directory**

   ```bash
   cd backend
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   uvicorn main:app --reload
   ```

5. **Access the application**
   Open your browser and navigate to `http://127.0.0.1:8000` to use the web interface.

### 🐳 Docker Usage

1. **Build the Docker image**
   Run this command from the **root** of the project (`ml-deployment/`):

   ```bash
   docker build -t breast-cancer-api -f backend/Dockerfile .
   ```

2. **Run the container**

   ```bash
   docker run -p 8000:8000 breast-cancer-api
   ```

3. **Access the application**
   Visit `http://localhost:8000` in your browser.

## 📡 API Endpoints

The API provides the following endpoints:

### 1. Web Interface

- **URL**: `/`
- **Method**: `GET`
- **Description**: Serves the HTML frontend.

### 2. Logistic Regression Prediction

- **URL**: `/predict/logistic`
- **Method**: `POST`
- **Body**:
  ```json
  {
    "features": [17.99, 10.38, 122.8, 1001.0, 0.1184]
  }
  ```
- **Response**:
  ```json
  {
    "model": "Logistic Regression",
    "prediction": 0
  }
  ```
  _(0 = Benign, 1 = Malignant)_

### 3. Decision Tree Prediction

- **URL**: `/predict/tree`
- **Method**: `POST`
- **Body**: Same as above.
- **Response**:
  ```json
  {
    "model": "Decision Tree",
    "prediction": 1
  }
  ```

## 📝 Input Features

The models expect the following 5 features in order:

1. Mean Radius
2. Mean Texture
3. Mean Perimeter
4. Mean Area
5. Mean Smoothness
