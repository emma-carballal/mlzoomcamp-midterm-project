import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

class HeartDisease(BaseModel):
    sex: str
    dataset: str
    cp: str
    fbs: str
    restecg: str
    exang: str
    slope: str
    age: int
    trestbps: float
    chol: float
    thalch: float
    oldpeak: float

model_ref = bentoml.sklearn.get("heart_disease_model:latest")
dv = model_ref.custom_objects["dictVectorizer"]
model_runner = model_ref.to_runner()

svc = bentoml.Service("heart_disease_classifier", runners=[model_runner])

@svc.api(input=JSON(pydantic_model=HeartDisease), output=JSON())
async def classify(heartdisease_application):
    application_data = heartdisease_application.dict()
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    result = prediction[0]

    if result == 0:
        return {"status" : "Normal heart"}
    else:
        return {"status": "Heart disease"}
