from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
import mlflow
import numpy as np

app = FastAPI()

host = "127.0.0.1"
port = "8080"

mlflow.set_tracking_uri(uri=f"http://{host}:{port}")

model_uri_format = "models:/{model_name}/{model_version}"

default_name = "tracking-quickstart"
default_version = 1
default_model_uri = model_uri_format.format(model_name=default_name, model_version=default_version)

model = mlflow.pyfunc.load_model(default_model_uri)

class PredictRequest(BaseModel):
    data: list[float]

class UpdateModelRequest(BaseModel):
    model_name: str
    model_version: str

@app.post("/predict")
async def predict(request: PredictRequest):
    if len(request.data) != 4:
        raise HTTPException(status_code=400, detail="Data must be of lenght 4")

    prediction = model.predict(np.array([request.data]))
    return {"y_pred": prediction.tolist()}

@app.post("/update-model")
async def update_model(request: UpdateModelRequest):
    global model
    print(request.model_name)
    print(request.model_version)
    model_uri = model_uri_format.format(model_name=request.model_name, model_version=request.model_version)
    model = mlflow.pyfunc.load_model(model_uri)
    return {"message": f"Modèle mis à jour vers {model_uri}"}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5027)
