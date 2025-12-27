import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

# ==============================
# 1. Load dataset
# ==============================
data = pd.read_csv("training/data/rentals.csv")

# ==============================
# 2. Features & Target
# ==============================
X = data[
    [
        "surface",          # m²
        "rooms",            # number of rooms
        "location_score",   # neighborhood attractiveness (1–10)
        "distance_center",  # km
        "season_index"      # seasonal demand (0.8–1.2)
    ]
]

y = data["price"]

# ==============================
# 3. Train / Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# 4. Model + Hyperparameter tuning
# ==============================
gbr = GradientBoostingRegressor(random_state=42)

param_grid = {
    "n_estimators": [100, 150],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4]
}

grid_search = GridSearchCV(
    gbr,
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# ==============================
# 5. Evaluation
# ==============================
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best parameters:", grid_search.best_params_)
print(f"MAE: {mae:.2f}")
print(f"R² score: {r2:.2f}")

# ==============================
# 6. Save model
# ==============================
joblib.dump(best_model, "app/models/price_model.pkl")
print("price_model.pkl saved successfully")
