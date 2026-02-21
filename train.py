# ===============================================
# 🌾 Professional Crop Yield Prediction Model
# ===============================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# ----------------------------------------
# 1. Load Data
# ----------------------------------------
df = pd.read_csv("Final_Dataset_after_temperature.csv")

print("Original Shape:", df.shape)

df = df.dropna()

# ----------------------------------------
# 2. Remove Data Leakage
# ----------------------------------------
df = df.drop(columns=["Production_in_tons"])

# ----------------------------------------
# 3. Remove Extreme Outliers (More Strict)
# ----------------------------------------
q_low = df["Yield_ton_per_hec"].quantile(0.02)
q_high = df["Yield_ton_per_hec"].quantile(0.98)

df = df[(df["Yield_ton_per_hec"] > q_low) &
        (df["Yield_ton_per_hec"] < q_high)]

print("After Cleaning Shape:", df.shape)

# ----------------------------------------
# 4. Log Transform Target (VERY IMPORTANT)
# ----------------------------------------
df["Yield_log"] = np.log1p(df["Yield_ton_per_hec"])

# ----------------------------------------
# 5. Feature Engineering
# ----------------------------------------
df["Rainfall_Temp"] = df["rainfall"] * df["temperature"]
df["Rainfall_sq"] = df["rainfall"] ** 2
df["Temp_sq"] = df["temperature"] ** 2
df["Area_log"] = np.log1p(df["Area_in_hectares"])

# ----------------------------------------
# 6. Split Features & Target
# ----------------------------------------
X = df.drop(["Yield_ton_per_hec", "Yield_log"], axis=1)
y = df["Yield_log"]

# Identify categorical & numeric columns
categorical_cols = ["State_Name", "Crop_Type", "Crop"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# ----------------------------------------
# 7. Preprocessing Pipeline
# ----------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ----------------------------------------
# 8. Model
# ----------------------------------------
model = XGBRegressor(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,          # L1 regularization
    reg_lambda=2,         # L2 regularization
    random_state=42,
    tree_method="hist"
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# ----------------------------------------
# 9. Train-Test Split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# ----------------------------------------
# 10. Evaluation
# ----------------------------------------
y_pred_log = pipeline.predict(X_test)

# Convert back from log scale
y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_test)

r2 = r2_score(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

print("\n===== FINAL PROFESSIONAL RESULTS =====")
print("R2 Score:", round(r2, 4))
print("RMSE:", round(rmse, 4))

# ----------------------------------------
# 11. Save Model
# ----------------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(pipeline, "models/crop_yield_pipeline.pkl")

print("\nModel Saved Successfully!")