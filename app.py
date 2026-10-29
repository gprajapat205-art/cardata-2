
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("car_price_model.pkl")
df = pd.read_csv("car data.csv")

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Used Car Price Predictor")
st.write("Enter the car details to estimate its selling price.")

# Input fields
car_name = st.selectbox(
    "Car Name",
    sorted(df["Car_Name"].unique())
)

year = st.selectbox(
    "Manufacturing Year",
    sorted(df["Year"].unique(), reverse=True)
)

present_price = st.number_input(
    "Present Showroom Price (in lakh)",
    min_value=0.0,
    value=5.0,
    step=0.1
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=30000,
    step=1000
)

fuel_type = st.selectbox(
    "Fuel Type",
    sorted(df["Fuel_Type"].unique())
)

seller_type = st.selectbox(
    "Seller Type",
    sorted(df["Seller_Type"].unique())
)

transmission = st.selectbox(
    "Transmission",
    sorted(df["Transmission"].unique())
)

owner = st.selectbox(
    "Previous Owners",
    sorted(df["Owner"].unique())
)

# Predict price
if st.button("Predict Selling Price", type="primary"):
    input_data = pd.DataFrame([{
        "Car_Name": car_name,
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
        "Owner": owner
    }])

    prediction = model.predict(input_data)[0]

    st.success("Estimated Selling Price")
    st.subheader(f"₹ {prediction:.2f} lakh")
    st.caption("This is an estimate, not a guaranteed market price.")
