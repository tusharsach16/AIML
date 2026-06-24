import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and metadata
model = joblib.load('house_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')
medians = joblib.load('feature_medians.pkl')

st.title("🏠 House Price Predictor")
st.write("Ridge Regression model — trained on Ames Housing dataset")

st.sidebar.header("Enter House Details")

# Sirf sabse important features input lo (poore 250 nahi — UX ke liye)
overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sqft)", 500, 5000, 1500)
total_bsmt_sf = st.sidebar.number_input("Basement Area (sqft)", 0, 3000, 800)
garage_cars = st.sidebar.slider("Garage Capacity (cars)", 0, 4, 2)
year_built = st.sidebar.number_input("Year Built", 1900, 2024, 2000)
full_bath = st.sidebar.slider("Full Bathrooms", 0, 4, 2)
neighborhood = st.sidebar.selectbox("Neighborhood", 
    ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt", "StoneBr", "Crawfor"])

if st.sidebar.button("Predict Price"):
    # Base row = medians (sab columns ke liye default value)
    input_data = pd.DataFrame([medians], columns=model_columns)
    
    # User inputs se overwrite karo (jo columns exist karte hain)
    if 'OverallQual' in input_data.columns:
        input_data['OverallQual'] = overall_qual
    if 'GrLivArea' in input_data.columns:
        input_data['GrLivArea'] = gr_liv_area
    if 'TotalBsmtSF' in input_data.columns:
        input_data['TotalBsmtSF'] = total_bsmt_sf
    if 'GarageCars' in input_data.columns:
        input_data['GarageCars'] = garage_cars
    if 'YearBuilt' in input_data.columns:
        input_data['YearBuilt'] = year_built
    if 'FullBath' in input_data.columns:
        input_data['FullBath'] = full_bath
    
    # Neighborhood one-hot column set karo
    neigh_col = f'Neighborhood_{neighborhood}'
    for col in input_data.columns:
        if col.startswith('Neighborhood_'):
            input_data[col] = 0
    if neigh_col in input_data.columns:
        input_data[neigh_col] = 1
    
    # Predict (log scale se actual price)
    pred_log = model.predict(input_data)[0]
    pred_price = np.expm1(pred_log)
    
    st.success(f"### Predicted Price: ${pred_price:,.0f}")
    
    st.write("---")
    st.write("**Model Limitations** (so you don't blindly trust it):")
    st.write("- Trained on Ames, Iowa data (2006-2010) — won't generalize to other markets")
    st.write("- Assumes linear relationships between features and price")
    st.write("- Doesn't account for current market conditions/inflation")