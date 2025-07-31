# model_tuning.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load saved splits
X_train = np.load("X_train.npy", allow_pickle=True)
X_test = np.load("X_test.npy", allow_pickle=True)
y_train = np.load("y_train.npy", allow_pickle=True)
y_test = np.load("y_test.npy", allow_pickle=True)

# --- Random Forest Hyperparameter Tuning ---
print("\n🔍 Tuning Random Forest...")

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='r2', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)

print("✅ Best Random Forest Params:", rf_grid.best_params_)
rf_best = rf_grid.best_estimator_

# --- XGBoost Hyperparameter Tuning ---
print("\n🔍 Tuning XGBoost...")

xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1],
}

xgb = XGBRegressor(random_state=42, verbosity=0)
xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='r2', n_jobs=-1, verbose=1)
xgb_grid.fit(X_train, y_train)

print("✅ Best XGBoost Params:", xgb_grid.best_params_)
xgb_best = xgb_grid.best_estimator_

# --- Evaluation ---
print("\n📊 Final Evaluation on Test Set:")
for name, model in {
    "Random Forest (Tuned)": rf_best,
    "XGBoost (Tuned)": xgb_best
}.items():
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n{name}:")
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.4f}")

    # Save model
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, filename)
