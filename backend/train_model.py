
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from .config import FEATURE_COLUMNS, TARGET_COLUMN, DATA_PATH, MODEL_PATH
from .utils import get_logger


def train_and_save_model():
    logger = get_logger("train_model")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Basic validation
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Handle missing values simply by filling with median
    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )

    logger.info("Starting model training...")
    pipeline.fit(X_train, y_train)
    logger.info("Model training finished.")

    # Evaluate
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"MAE: {mae:.3f}, R2: {r2:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    print(f"Training done. MAE={mae:.3f}, R2={r2:.3f}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
