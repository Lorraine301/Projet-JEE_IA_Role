import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Charger les données locataires
data = pd.read_csv("training/data/tenants.csv")

# Features & label
X = data[[
    "late_payments",      # nombre de retards
    "disputes",           # nombre de litiges
    "rental_duration"     # durée moyenne (mois)
]]

y = data["default"]       # 1 = risqué, 0 = fiable

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modèle
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Sauvegarde
joblib.dump(model, "app/models/risk_model.pkl")
print("risk_model.pkl saved")
