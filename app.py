import streamlit as st
import joblib
import pandas as pd

# Load the model and preprocessor
model = joblib.load('car_price_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Streamlit app
st.set_page_config(page_title="Car Price Prediction App", page_icon="🚗")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
    
        color: #fff; /* Text color */
    }
    .title {
        text-align: center;
        font-size: 2em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .stNumberInput, .stSelectbox, .stButton {
        background-color: rgba(255, 255, 255, 0.8); /* Light background for input fields */
        color: #000; /* Input text color */
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<h1 class="title">Car Price Prediction App</h1>', unsafe_allow_html=True)

# Collect user input
year = st.number_input("Year of the Car", min_value=1990, max_value=2024, value=2015)
present_price = st.number_input("Present Price of the Car (in lakhs)", min_value=0.0, value=5.0)
driven_kms = st.number_input("Kilometers Driven", min_value=0, value=20000)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
selling_type = st.selectbox("Selling Type", ['Dealer', 'Individual'])
transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'])
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
Car_Name = st.selectbox("Car Name", ['ritz', 'sx4', 'ciaz', 'wagon r', 'swift', 'vitara brezza', 's cross',
                                       'alto k10', 'ertiga', 'dzire', 'ignis', '800'])

# Button to predict price
if st.button("Predict Price"):
    # Convert inputs into a DataFrame for the preprocessor
    input_data = pd.DataFrame([[year, Car_Name, present_price, driven_kms, fuel_type, selling_type, transmission, owner]],
                              columns=['Year', 'Car_Name', 'Present_Price', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner'])

    # Debugging: Check the columns
    st.write("Input Data Columns:", input_data.columns)

    # Preprocess the input data
    processed_data = preprocessor.transform(input_data)

    # Get prediction
    prediction = model.predict(processed_data)

    st.write(f"Estimated Selling Price: {prediction[0]:.2f} lakhs")
