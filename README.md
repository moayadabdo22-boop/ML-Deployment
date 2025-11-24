# Task3 – End-to-End Machine Learning Deployment (California Housing Price Prediction)

This project demonstrates a complete end-to-end machine learning pipeline: training a regression model, saving it, exposing it through an API, and interacting with it via a web interface. The model predicts housing prices in California based on input features such as population, number of rooms, median income, and more. The web interface allows users to upload a CSV file, receive predictions, and view a generated histogram plot.

## Project Structure
Task3-ML  
│  
├─ backend → FastAPI backend files (`app.py`, `train_model.py`, `preprocess.py`, `predict.py`, `visualize.py`, `config.py`, `utils.py`, static folder)  
├─ data → dataset (housing.csv)  
├─ models → saved machine learning model (model.joblib)  
├─ frontend → index.html web page  
├─ venv → virtual environment  
└─ requirements.txt → dependencies  

## Features
- Train a regression model using scikit-learn and save it with joblib  
- Preprocess uploaded CSV data and validate feature columns  
- FastAPI backend with `/predict` and `/health` endpoints  
- Generate histogram visualization of predicted values  
- Frontend user interface for uploading CSV and viewing predictions  

## Technologies Used
scikit-learn, pandas, matplotlib, joblib, FastAPI, Uvicorn, HTML + JavaScript

## How to Run the Project
1. Activate virtual environment  
2. Install dependencies  
3. Train the model  
4. Start FastAPI server  
API URL: `http://127.0.0.1:8000`  
Swagger docs: `http://127.0.0.1:8000/docs`

5. Open the web interface  
Open the file `frontend/index.html` manually in a browser. Upload the CSV file to get predictions and the histogram plot.

## CSV Requirements
The CSV file must include the following columns:  

## Output
After uploading the CSV file from the web page, the system returns:  
- Number of processed rows  
- Prediction table (first 20 rows)  
- Histogram plot of predicted house values  

## Notes
- Retraining the model overwrites the file `model.joblib`  
- If column names in the dataset change, update `FEATURE_COLUMNS` in `config.py`  

## Author
Moayad Rabah
