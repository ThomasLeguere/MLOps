from fastapi import FastAPI
import joblib
import uvicorn

app = FastAPI()


@app.post("/predict")
async def read_root(size: int, nb_rooms: int, garden: bool):
    if (size < 0 or nb_rooms < 0):
        return {"y_pred": None}
    
    regression_model = joblib.load("regression.joblib")
    prediction = regression_model.predict([[size, nb_rooms, garden]])
    return {"y_pred": prediction[0]}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5027)
