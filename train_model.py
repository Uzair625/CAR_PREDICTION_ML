"""
Model training pipeline — works with both the original India dataset and the
merged global dataset (data/processed/global_cars.csv).

Run directly:          python train_model.py
Called by collection:  train_model.run(data_path='...')
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

MODEL_OUT   = 'model.pkl'
RESULTS_OUT = 'model_results.pkl'


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(df)} rows from {data_path}")
    return df


def prepare_features(df: pd.DataFrame):
    """
    Detect whether we have the original India schema (Price column)
    or the global schema (Price_USD column) and build X, y accordingly.
    """
    # --- Target ---
    if 'Price_USD' in df.columns:
        y = pd.to_numeric(df['Price_USD'], errors='coerce')
        is_global = True
    elif 'Price' in df.columns:
        y = pd.to_numeric(df['Price'], errors='coerce')
        is_global = False
    else:
        raise ValueError("Dataset must have a 'Price' or 'Price_USD' column.")

    # --- Categorical features ---
    cat_cols = ['name', 'company', 'fuel_type']
    if is_global and 'country' in df.columns:
        cat_cols.append('country')
    if 'transmission' in df.columns:
        cat_cols.append('transmission')

    # --- Numerical features ---
    num_cols = ['year', 'kms_driven']

    # --- Build X ---
    all_feature_cols = cat_cols + num_cols
    available = [c for c in all_feature_cols if c in df.columns]
    X = df[available].copy()

    # Ensure numerical cols are numeric
    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce')

    # Fill missing categoricals with 'Unknown'
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].fillna('Unknown').astype(str)

    # Drop rows where target or year is missing
    mask = y.notna() & X['year'].notna()
    X, y = X[mask], y[mask]

    actual_cat = [c for c in cat_cols if c in X.columns]
    actual_num = [c for c in num_cols if c in X.columns]

    return X, y, actual_cat, actual_num


def build_pipeline(cat_cols, num_cols, model):
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', num_transformer, num_cols),
    ])
    return Pipeline([('preprocessor', preprocessor), ('model', model)])


def run(data_path: str = None):
    # Auto-detect best available dataset
    if data_path is None:
        global_path = os.path.join('data', 'processed', 'global_cars.csv')
        data_path   = global_path if os.path.exists(global_path) else 'cleaned_car.csv'

    df = load_data(data_path)
    X, y, cat_cols, num_cols = prepare_features(df)

    print(f"Features (cat): {cat_cols}")
    print(f"Features (num): {num_cols}")
    print(f"Training samples: {len(X)}")
    print(f"Price range: ${y.min():,.0f} - ${y.max():,.0f}")

    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'Ridge Regression':      Ridge(alpha=10),
        'Random Forest':         RandomForestRegressor(n_estimators=200, random_state=42,
                                                        min_samples_leaf=3, n_jobs=-1),
        'Gradient Boosting':     GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                                            max_depth=5, random_state=42,
                                                            subsample=0.8),
        'Linear Regression':     LinearRegression(),
    }

    results      = {}
    best_pipeline = None
    best_score    = -np.inf
    best_name     = ''

    print("\n--- Model Evaluation (test set) ---")
    for name, model in models.items():
        pipe = build_pipeline(cat_cols, num_cols, model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        r2  = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {'r2': r2, 'mae': mae}
        print(f"  {name:<28}: R2={r2:.4f}  MAE=${mae:,.0f}")

        if r2 > best_score:
            best_score    = r2
            best_pipeline = pipe
            best_name     = name

    print(f"\nBest model: {best_name}  (R2={best_score:.4f})")
    print(f"Dataset source: {data_path}")

    # Save
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(best_pipeline, f)

    meta = {
        'results':    results,
        'best_name':  best_name,
        'best_score': best_score,
        'cat_cols':   cat_cols,
        'num_cols':   num_cols,
        'data_path':  data_path,
        'n_samples':  len(X),
    }
    with open(RESULTS_OUT, 'wb') as f:
        pickle.dump(meta, f)

    print(f"Saved: {MODEL_OUT}, {RESULTS_OUT}")
    return best_pipeline, meta


if __name__ == '__main__':
    run()
