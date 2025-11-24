
import os
from typing import Sequence
import matplotlib.pyplot as plt

from .config import PLOT_PATH
from .utils import get_logger

logger = get_logger("visualize")


def create_predictions_plot(predictions: Sequence[float], out_path: str = PLOT_PATH):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.hist(predictions, bins=30)
    plt.title("Predicted House Values")
    plt.xlabel("Predicted value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    logger.info(f"Prediction plot saved to {out_path}")
    return out_path
