# A heart disease classifier - A DataTalks ML Zoomcamp Midterm Project

This is a midterm project for the [DataTalks ml-zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp).

I selected and trained a model for binary classification that predicts the presence or absence of heart disease based on a series of attributes.
The dataset is a subset of the Heart Disease Data Set from UCI Machine Learning data repository and can be downloaded from [kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data/download?datasetVersionNumber=6)

The data preparation, EDA, model selection and fine-tuning process are contained in the `notebook.ipynb` in the `notebook` folder.

The model has been saved, built and dockerized with BentoML.
Te prediction API has been deployed on AWS Elastic Container Service and you can test [here](http://54.152.57.142:3000/). You will find an example of a data query in the `predict_test.py` file.

## How to run the project
### Training and saving the model.
