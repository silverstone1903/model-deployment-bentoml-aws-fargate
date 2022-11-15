import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
import bentoml

warnings.filterwarnings("ignore")
pd.options.display.max_columns = 100

seed = 2022

data = pd.read_csv("data/HR-Employee-Attrition.csv")
data["Attrition"] = data["Attrition"].map({'Yes': 1, 'No': 0})
constant_cols = data.nunique()[data.nunique() == 1].keys().tolist()
data.drop(constant_cols, axis=1, inplace=True)
target = ["Attrition"]
num_cols = [
    num for num in data.select_dtypes(exclude=["O"]).columns.tolist()
    if num not in target + ["EmployeeNumber"] + constant_cols
]
cat_cols = [
    cat for cat in data.select_dtypes("O").columns.tolist()
    if cat not in constant_cols
]
scale_cols = [
    c for c in data.nunique()[data.nunique() > 10].keys().tolist()
    if c not in ["EmployeeNumber"] + target
]

df = data.copy()
df = df.replace([np.inf, -np.inf], np.nan)

for cat in cat_cols:
    le = LabelEncoder()
    df[cat] = le.fit_transform(df[cat])

train_cols = [
    'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender'
]

x_train, x_test, y_train, y_test = train_test_split(df[train_cols],
                                                    df[target],
                                                    test_size=0.2,
                                                    random_state=2020,
                                                    stratify=df[target])

model = RandomForestClassifier(n_estimators=150,
                               class_weight="balanced",
                               min_samples_leaf=10,
                               min_samples_split=5,
                               random_state=seed)
model.fit(x_train, y_train)
model_preds = model.predict_proba(x_test)[:, 1]

acc = accuracy_score(y_test, np.round(model_preds))
print("Accuracy score: ", acc)

print('\nSaving model as a Bento')
bentoml.sklearn.save_model('hr_attrition', model, metadata={"acc": acc})
