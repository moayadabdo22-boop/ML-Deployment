# Task 3 – End-to-End Machine Learning Deployment (California Housing Price Prediction)

This project demonstrates a complete end-to-end ML deployment pipeline: preprocessing data, training a regression model on the California Housing dataset, exposing predictions through a FastAPI backend, and visualizing results through a web-based frontend. Users upload a CSV file containing input housing features, and the system returns price predictions along with a visualization plot.

## System Overview

| Component | Description |
|----------|-------------|
| Machine Learning Model | Random Forest Regressor trained on California Housing dataset |
| Backend API | FastAPI + Uvicorn |
| Visualization | Matplotlib plot of predicted values |
| Frontend | HTML + CSS + JavaScript (single-page interface) |
| Dataset | housing.csv |
| Environment | WSL2 (Ubuntu) + Python virtual environment |

## Project Structure
Task3-ML
│
├─ backend/ # ML logic + API
│ ├─ app.py # FastAPI application + /predict endpoint
│ ├─ train_model.py # Train and save model
│ ├─ preprocess.py # Data validation and preprocessing
│ ├─ predict.py # Inference helper functions
│ ├─ visualize.py # Generates plot from predictions
│ ├─ config.py # Constants and paths
│ ├─ utils.py # Shared utilities
│ └─ init.py
│
├─ data/
│ └─ housing.csv # Dataset used for training and testing
│
├─ models/
│ └─ .gitkeep # Model file excluded due to GitHub size limit
│
├─ frontend/
│ └─ index.html # Single-page UI for file upload and visualization
│
├─ logs.txt # Logging for training and debugging
├─ requirements.txt # Python dependencies
├─ .gitignore # Prevents committing venv/model files
└─ README.md # Project documentation


> The trained model file (model.pkl) is not included on GitHub due to the platform’s 100MB size limit. It is automatically generated when running `train_model.py`.

## Dataset Feature Requirements

The CSV file used for prediction must contain the following columns:

| Feature | Meaning |
|--------|---------|
| MedInc | Median income in block |
| HouseAge | Average house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Block population |
| AveOccup | Average occupants per household |
| Latitude | Geographical coordinate |
| Longitude | Geographical coordinate |

Any missing or non-numeric values are handled automatically during preprocessing.

## How to Run the Project Locally

### 1) Activate virtual environment


cd Task3-ML
source venv/bin/activate


### 2) Install dependencies


pip install -r requirements.txt


### 3) Train the machine learning model


python3 backend/train_model.py


### 4) Start the FastAPI server


uvicorn backend.app:app --reload


### 5) Launch the frontend
Open the file below manually in a browser:


frontend/index.html


After uploading the CSV file, the UI will show:
- Number of processed rows
- First 20 predictions
- Plot visualization of predicted values

## Notes

- Retraining the model overwrites `models/model.pkl`
- If dataset column names change, update `FEATURE_COLUMNS` in `config.py`
- Logging information can be checked in `logs.txt`

## Author

**Moayad Rabah**

GitHub Repository:  
https://github.com/moayadabdo22-boop/ML-Deployment
