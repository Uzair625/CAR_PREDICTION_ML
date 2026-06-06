import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os

# Load data
df = pd.read_csv('cleaned_car.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nSample:\n{df.head()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Drop rows with missing values
df = df.dropna()

# Feature engineering: car age from year
current_year = 2024
df['car_age'] = current_year - df['year']

# Features and target
X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

print(f"\nTraining on {len(X)} samples")
print(f"Price range: Rs.{y.min():,.0f} - Rs.{y.max():,.0f}")

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
categorical_features = ['name', 'company', 'fuel_type']
numerical_features = ['year', 'kms_driven']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# Train and evaluate multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1),
}

results = {}
best_pipeline = None
best_score = -np.inf
best_name = ''

print("\n--- Model Evaluation ---")
for model_name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results[model_name] = {'r2': r2, 'mae': mae}
    print(f"{model_name:25s}: R2 = {r2:.4f}  |  MAE = Rs.{mae:,.0f}")

    if r2 > best_score:
        best_score = r2
        best_pipeline = pipeline
        best_name = model_name

print(f"\nBest model: {best_name} (R² = {best_score:.4f})")

# Save the best pipeline (includes preprocessor + model)
with open('model.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)

# Save model results for display in app
with open('model_results.pkl', 'wb') as f:
    pickle.dump({'results': results, 'best_name': best_name, 'best_score': best_score}, f)

print("\nSaved: model.pkl, model_results.pkl")
