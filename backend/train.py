from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load data
data = load_breast_cancer(as_frame=True)
X = data.data[
    ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]
]
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train models
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(random_state=42)

lr.fit(X_train_scaled, y_train)
dt.fit(X_train, y_train)

# Save models
joblib.dump(lr, MODEL_DIR / "logistic_regression.joblib")
joblib.dump(dt, MODEL_DIR / "decision_tree.joblib")
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

print("Models trained and saved successfully")
