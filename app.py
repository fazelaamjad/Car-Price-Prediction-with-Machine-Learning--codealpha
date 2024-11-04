import streamlit as st
import joblib

# Load the model
model = joblib.load('car_price_model.pkl')

# Streamlit app
st.title("Car Price Prediction App")

# Collect user input
year = st.number_input("Year of the Car", min_value=1990, max_value=2024, value=2015)
present_price = st.number_input("Present Price of the Car (in lakhs)", min_value=0.0, value=5.0)
driven_kms = st.number_input("Kilometers Driven", min_value=0, value=20000)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
selling_type = st.selectbox("Selling Type", ['Dealer', 'Individual'])
transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'])
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

# Button to predict price
if st.button("Predict Price"):
    # Convert inputs into the format required by the model
    features = [[year, present_price, driven_kms, fuel_type, selling_type, transmission, owner]]

    # Get prediction
    prediction = model.predict(features)

    st.write(f"Estimated Selling Price: {prediction[0]} lakhs")
