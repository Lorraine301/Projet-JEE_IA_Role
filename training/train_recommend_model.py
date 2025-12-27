import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Charger données logements
data = pd.read_csv("training/data/properties.csv")

# Features utilisateurs / biens
X = data[[
    "price",
    "surface",
    "rooms",
    "location_score",
    "lifestyle_score"
]]

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
kmeans = KMeans(
    n_clusters=5,
    random_state=42
)
kmeans.fit(X_scaled)

# Sauvegarde modèle + scaler
joblib.dump(kmeans, "app/models/recommend_model.pkl")
joblib.dump(scaler, "app/models/recommend_scaler.pkl")

print("recommend_model.pkl saved")
