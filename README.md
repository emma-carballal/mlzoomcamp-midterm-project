# A heart disease classifier - A DataTalks ML Zoomcamp Midterm Project

This is a midterm project for the [DataTalks ml-zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp).

I selected and trained a model for binary classification that predicts the presence or absence of heart disease based on a series of attributes.
The dataset is a subset of the Heart Disease Data Set from UCI Machine Learning data repository and can be downloaded from [kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/download?datasetVersionNumber=6)

The data preparation, EDA, model selection and fine-tuning process is contained in the `notebook.ipynb` in the `notebook` folder.

The model has been saved, built and dockerized with [BentoML](https://github.com/bentoml/BentoML), an open-source library that enables users to create a machine learning-powered prediction service easily.

The prediction API has been deployed on AWS Elastic Container Service and you can test it [here](http://54.152.57.142:3000/). An example of a json data query is provided in the `predict_test.py` file.

## How to run the project
### Train and save the model
Create and activate an environment for the project, for example with with anaconda.

```conda create --name heartdisease```

Install BentoML in your environment together with the dependencies required to run this model.

```pip install bentoml scikit-learn pandas pydantic```

To train the model, execute the ``train.py`` file.

```python train.py```

The model will be saved in a local directory managed by BentoML where it will be fetched to build the bento.

### Build and serve a bento
In order to serve an API, build the service and the model into a bento.

```bentoml build```

Serve the bento with the service locally.

```bentoml serve heart_disease_classifier:latest --production```

And access the API on http://localhost:3000

Test the service API in the "Try it out" section, pasting the following data in the request body:

```
{
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
```

Is this patient at risk of heart disease?
