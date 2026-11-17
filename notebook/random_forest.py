import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib


df = pd.read_csv("AmesHousing.csv")

features = [
    "Overall Qual", 
    "Gr Liv Area", 
    "Garage Cars", 
    "Total Bsmt SF", 
    "Full Bath", 
    "Year Built"
]

X = df[features]
y = np.log1p(df["SalePrice"])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])


model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"RMSE: ${rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

joblib.dump(model, "house_price_pipeline.pkl")