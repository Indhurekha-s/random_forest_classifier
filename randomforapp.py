import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Load trained model
# ---------------------------
with open("randomforestclassifier.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Page title
# ---------------------------
st.set_page_config(page_title="Credit Card Fraud Detection")
st.title("💳 Credit Card Fraud Detection App")

st.write("Enter transaction details to predict Fraud or Not Fraud")

# ---------------------------
# User inputs
# ---------------------------
transaction_id = st.number_input("Transaction ID", min_value=1, step=1)
amount = st.number_input("Transaction Amount", min_value=0.0, step=1.0)
merchant_id = st.number_input("Merchant ID", min_value=1, step=1)

transaction_type = st.selectbox(
    "Transaction Type",
    ["purchase", "refund"]
)

location = st.selectbox(
    "Location",
    ["San Antonio", "Philadelphia", "Houston", "New York", "Chicago"]
)

# ---------------------------
# Encode categorical values
# (same logic as training)
# ---------------------------
le_type = LabelEncoder()
le_loc = LabelEncoder()

le_type.fit(["purchase", "refund"])
le_loc.fit(["San Antonio", "Philadelphia", "Houston", "New York", "Chicago"])

transaction_type_encoded = le_type.transform([transaction_type])[0]
location_encoded = le_loc.transform([location])[0]

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Fraud"):
    input_data = pd.DataFrame([[
        transaction_id,
        amount,
        merchant_id,
        transaction_type_encoded,
        location_encoded
    ]], columns=[
        "TransactionID",
        "Amount",
        "MerchantID",
        "TransactionType",
        "Location"
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Transaction Detected! (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Transaction is Safe (Probability: {1 - probability:.2f})")
