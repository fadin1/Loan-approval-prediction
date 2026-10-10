
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate sample data
np.random.seed(42)
n = 2000

data = pd.DataFrame({
    "Gender": np.random.choice(
        ["Male", "Female"], n
    ),
    "Married": np.random.choice(
        ["Yes", "No"], n
    ),
    "Dependents": np.random.choice(
        ["0", "1", "2", "3+"], n
    ),
    "Education": np.random.choice(
        ["Graduate", "Not Graduate"], n
    ),
    "Self_Employed": np.random.choice(
        ["Yes", "No"], n
    ),
    "ApplicantIncome": np.random.randint(
        1500, 15001, n
    ),
    "CoapplicantIncome": np.random.randint(
        0, 8001, n
    ),
    "LoanAmount": np.random.randint(
        30, 601, n
    ),
    "Loan_Amount_Term": np.random.choice(
        [120, 180, 240, 300, 360], n
    ),
    "Credit_History": np.random.choice(
        [0.0, 1.0], n, p=[0.2, 0.8]
    ),
    "Property_Area": np.random.choice(
        ["Urban", "Semiurban", "Rural"], n
    ),
})

# Create synthetic labels for demonstration.
# These are NOT real bank approval decisions.
income = (
    data["ApplicantIncome"]
    + data["CoapplicantIncome"]
)

loan = data["LoanAmount"]

score = (
    2.5 * (data["Credit_History"] == 1).astype(int)
    + 0.5 * (data["Education"] == "Graduate").astype(int)
    + 0.3 * (data["Property_Area"] == "Semiurban").astype(int)
    + 0.3 * (data["Married"] == "Yes").astype(int)
    + 0.4 * (income / (loan * 100 + 1))
    - 1.0
    + np.random.normal(0, 0.8, n)
)

data["Loan_Status"] = np.where(score > 0.7, "Y", "N")

# Separate features and target
X = data.drop(columns=["Loan_Status"])
y = data["Loan_Status"]

categorical_features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

numeric_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features),
])

# Create model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Train
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("Model training completed!")
print("Test accuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions))

# Save model and expected feature order
joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
    },
    "loan_model.pkl",
)

print("\nModel saved as loan_model.pkl")