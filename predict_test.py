import requests

data={
    "sex": "Female",
    "dataset": "Hungary",
    "cp": "atypical angina",
    "fbs": "False",
    "restecg": "st-t abnormality",
    "exang": "False",
    "slope": "flat",
    "age": 31,
    "trestbps": 100.0,
    "chol": 219.0,
    "thalch": 150.0,
    "oldpeak": 0.0
}

result = requests.post(
   "http://localhost:3000/classify",
   headers={"content-type": "application/json"},
   json=data,
).json()

print(result)
