import mlflow
from mlflow.models import infer_signature

host = "127.0.0.1"
port = "8000"
mlflow.set_tracking_uri(uri=f"http://{host}:{port}")

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model("models:/{model_name}/{model_version}".format(model_name="tracking-quickstart", model_version=4))

predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions
print(result[:4])

