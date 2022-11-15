import numpy as np

import bentoml
from bentoml.io import JSON

model = bentoml.sklearn.get("hr_attrition").to_runner()
svc = bentoml.Service("hr_attrition_model", runners=[model])


@svc.api(input=JSON(), output=JSON())
def classify(df):
    vector = np.array([[
        df['BusinessTravel'], df['DailyRate'], df['Department'],
        df['DistanceFromHome'], df['Education'], df['EducationField'],
        df['EnvironmentSatisfaction'], df['Gender']
    ]])
    prediction = model.predict.run(vector)
    pred = prediction[0]

    if pred == 0:
        return {"Attrition": "No"}
    elif pred == 1:
        return {"Attrition": "Yes"}
