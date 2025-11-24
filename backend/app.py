
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import pandas as pd
from io import StringIO

from .preprocess import validate_and_preprocess
from .predict import load_model, make_predictions
from .visualize import create_predictions_plot
from .config import PLOT_PATH
from .utils import get_logger

logger = get_logger("app")

# VERY IMPORTANT: this name must be exactly "app"
app = FastAPI(title="California Housing Prediction API")

# Enable CORS (for local frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (prediction plot)
app.mount("/static", StaticFiles(directory="backend/static"), name="static")


@app.get("/health")
def health_check():
    return {"status": "ok"}
@app.get("/")
def root():
    return {"message": "API is running. Go to /docs or use the frontend page."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a CSV file, run preprocessing and prediction,
    return predictions and plot URL.
    """
    try:
        if file.content_type not in ("text/csv", "application/vnd.ms-excel"):
            raise HTTPException(status_code=400, detail="Please upload a CSV file.")

        contents = await file.read()

        # Decode bytes -> string, then read CSV
        try:
            df = pd.read_csv(StringIO(contents.decode("utf-8")))
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            raise HTTPException(status_code=400, detail="Could not parse CSV file.")

        # Preprocess features
        try:
            X = validate_and_preprocess(df)
        except ValueError as ve:
            logger.error(str(ve))
            raise HTTPException(status_code=400, detail=str(ve))

        # Load model and predict
        model = load_model()
        preds = make_predictions(model, X)

        # Create plot image
        plot_path = create_predictions_plot(preds.values, out_path=PLOT_PATH)
        plot_url = "/static/" + plot_path.split("/")[-1]

        return {
            "num_rows": len(preds),
            "predictions": preds.tolist(),
            "plot_url": plot_url,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in /predict: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
