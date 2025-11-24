
import pandas as pd
from typing import List
from .config import FEATURE_COLUMNS
from .utils import get_logger


logger = get_logger("preprocess")


def validate_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that all required feature columns exist and preprocess the data:
    - keep only FEATURE_COLUMNS
    - fill missing numeric values with column median
    """
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        msg = f"Uploaded file is missing required columns: {missing_cols}"
        logger.error(msg)
        raise ValueError(msg)

    # Keep only required columns in correct order
    X = df[FEATURE_COLUMNS].copy()

    # Fill missing numeric values with column median
    X = X.fillna(X.median(numeric_only=True))

    return X
