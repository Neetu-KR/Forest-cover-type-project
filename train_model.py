import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("train.csv")

# Feature engineering
df["Distance_To_Hydrology"] = np.sqrt(
    df["Horizontal_Distance_To_Hydrology"]**2 +
    df["Vertical_Distance_To_Hydrology"]**2
)

df["Terrain_Ruggedness"] = df["Elevation"] * df["Slope"]

# Prepare data
X = df.drop(["Id", "Cover_Type"], axis=1)
y = df["Cover_Type"]

# Save columns and median values
joblib.dump(X.columns.tolist(), "model_columns.pkl")
joblib.dump(X.median(), "median_values.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 1. Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

# 2. Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)
gb_model.fit(X_train, y_train)

# 3. KNN
knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
knn_model.fit(X_train, y_train)

# Save models
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(gb_model, "gradient_boosting_model.pkl")
joblib.dump(knn_model, "knn_model.pkl")

# Accuracy
print("Random Forest Accuracy:",
      rf_model.score(X_test, y_test))

print("Gradient Boosting Accuracy:",
      gb_model.score(X_test, y_test))

print("KNN Accuracy:",
      knn_model.score(X_test, y_test))

print("\nAll 3 models trained and saved successfully!")