
import os
from typing import Tuple
import pandas as pd
import joblib

from .config import MODEL_PATH
from .utils import get_logger

logger = get_logger("predict")


def load_model():
    if not os.path.exists(MODEL_PATH):
        msg = f"Model file not found at {MODEL_PATH}. Train the model first."
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Loading model...")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded.")
    return model


def make_predictions(model, X: pd.DataFrame) -> pd.Series:
    logger.info(f"Making predictions for {len(X)} rows.")
    preds = model.predict(X)
    return pd.Series(preds, index=X.index, name="predicted_price")
