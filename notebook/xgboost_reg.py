import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
from sklearn.metrics import mean_squared_error, r2_score


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

# Define Monotonic Constraints for XGBoost
constraints = (1, 1, 1, 1, 1, 1)


model = XGBRegressor(
    n_estimators=150,     
    learning_rate=0.1,     
    max_depth=3,           
    monotone_constraints=constraints,
    random_state=42
)

print("Training XGBoost model...")
model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"✅ XGBoost RMSE: ${rmse:,.2f}")
print(f"✅ XGBoost R2 Score: {r2:.4f}")

joblib.dump(model, "house_price_pipeline.pkl")
print("Model saved successfully as house_price_pipeline.pkl!")