
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import bentoml


#Loading the dataset

df = pd.read_csv("heart_disease_uci.csv")

# Imputing NaNs

df["fbs"].fillna(False, inplace=True)
df["restecg"].fillna("normal", inplace=True)
df["exang"].fillna(False, inplace=True)
df["slope"].fillna("flat", inplace=True)

numerical_nans = ["trestbps", "chol", "thalch", "oldpeak"]
for col in numerical_nans:
    median = df[col].median()
    df[col].fillna(median, inplace=True)

# Deleting unwanted features

del df["id"]
del df["ca"]
del df["thal"]

# Turning target column to binary values

df.num.loc[df.num != 0] = 1

# Changing target column name to something more descriptive.

df["disease"] = df.num
df.drop("num", axis=1, inplace=True)

# Setting up the validation framework

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.disease.values
y_test = df_test.disease.values

df_train.drop("disease", axis=1, inplace=True)
df_test.drop("disease", axis=1, inplace=True)

# Training and validation dataframes are vectorized
# in order to encode categorical features.

categorical = ["sex", "dataset", "cp", "fbs", "restecg", "exang", "slope"]
numerical = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']

train_dict = df_train[categorical + numerical].to_dict(orient='records')
test_dict = df_test[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dict)
X_test = dv.transform(test_dict)

# Final model is trained (model selection and finetuning explained in notebook.ipynb)

model = LogisticRegression(max_iter=1500, solver='liblinear', penalty='l1')
model.fit(X_train, y_train)

# Saving the model

bentoml.sklearn.save_model('heart_disease_model', model, custom_objects={"dictVectorizer": dv})

print("Logistic regression model has been trained and saved to the model store (a local directory managed by BentoML) with the name heart_disease_model")
