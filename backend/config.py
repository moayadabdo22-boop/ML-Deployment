
FEATURE_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

TARGET_COLUMN = "median_house_value"

DATA_PATH = "data/housing.csv"
MODEL_PATH = "models/model.pkl"
PLOT_PATH = "backend/static/predictions.png"
LOG_PATH = "logs.txt"
