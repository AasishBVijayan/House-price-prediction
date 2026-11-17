# House Price Prediction

A machine learning-powered web application for estimating residential property values based on key housing attributes such as square footage, quality, garage capacity, basement size, bathrooms, and year built.

This project combines a trained regression model with an interactive Streamlit interface, making it easy to explore property valuation in real time.

## Demo

![House Price Prediction App](screenshots/house%20price.png)

## Project Overview

The goal of this project is to predict a home's sale price using a curated subset of features from the Ames Housing dataset. The application loads a serialized machine learning pipeline and estimates a property value from user inputs.

The system is designed for:
- Quick residential valuation
- Interactive experimentation with property attributes
- Model-backed predictions without writing code
- A simple local deployment for demos or learning purposes

## Features

- User-friendly Streamlit dashboard
- Real-time house price estimation
- Input controls for the most influential property variables
- Model inference using a trained ML pipeline
- Automatic reverse transformation from log-price output to actual USD value
- Clean and lightweight deployment setup

## Model Inputs

The prediction model uses the following features:

- Overall Quality
- Ground Living Area (sq.ft)
- Garage Capacity (cars)
- Basement Area (sq.ft)
- Full Bathrooms
- Year Built

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- XGBoost
- Joblib

## Project Structure

```text
House-price-prediction/
├── app.py                       # Streamlit application
├── house_price_pipeline.pkl    # Trained ML pipeline used for predictions
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── notebook/
│   ├── AmesHousing.csv         # Housing dataset
│   ├── eda.ipynb              # Exploratory data analysis notebook
│   ├── linear_reg.py          # Linear regression training script
│   ├── main.ipynb             # Main notebook workflow
│   ├── random_forest.py       # Random forest training script
│   └── xgboost_reg.py         # XGBoost model training script
└── screenshots/
    └── house price.png         # Application preview
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/AasishBVijayan/House-price-prediction.git
cd House-price-prediction
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

## How It Works

1. The user enters property details into the Streamlit form.
2. The input is converted into a pandas DataFrame matching the model's expected feature names.
3. The saved pipeline predicts the log-transformed sale price.
4. The value is converted back to original price scale using `expm1`.
5. The final estimated property value is displayed in USD.

## Model Training Notes

Several training scripts were explored in the `notebook/` folder:

- `linear_reg.py` — baseline linear regression model
- `random_forest.py` — ensemble-based regression approach
- `xgboost_reg.py` — boosted tree model with structured feature handling

The final deployment model is stored in `house_price_pipeline.pkl` and is loaded directly by the app.

## Example Prediction

A typical property estimate might look like this:

- Overall Quality: 7
- Living Area: 2,100 sq.ft
- Garage Capacity: 2
- Basement Area: 1,200 sq.ft
- Full Bathrooms: 2
- Year Built: 2004

Estimated Value: `$320,000+`

Exact results depend on the trained model and the dataset distribution used during training.

## Author

Aasish B Vijayan

## Acknowledgments

This project uses the Ames Housing dataset and demonstrates a practical end-to-end machine learning workflow from data preparation and model training to deployment in a web application.
