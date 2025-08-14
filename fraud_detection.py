import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline model
try:
    model = joblib.load("fraud_detection_pipeline.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("FRAUD DETECTION PREDICTION APP")
st.markdown("Please enter the transaction details and use the predict button")
st.divider()

# Input fields
col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "Transaction Type", 
        ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    )
    amount = st.number_input(
        "Amount", 
        min_value=0.0, 
        value=1000.0,
        help="The amount of the transaction"
    )
    oldbalanceOrg = st.number_input(
        "Old Balance (Sender)", 
        min_value=0.0, 
        value=10000.0,
        help="Sender's balance before transaction"
    )
    
with col2:
    newbalanceOrig = st.number_input(
        "New Balance (Sender)", 
        min_value=0.0, 
        value=9000.0,
        help="Sender's balance after transaction"
    )
    oldbalanceDest = st.number_input(
        "Old Balance (Receiver)", 
        min_value=0.0, 
        value=0.0,
        help="Receiver's balance before transaction"
    )
    newbalanceDest = st.number_input(
        "New Balance (Receiver)", 
        min_value=0.0, 
        value=0.0,
        help="Receiver's balance after transaction"
    )

# Prediction button
if st.button("Predict Fraud"):
    # Create input DataFrame with the same structure as training data
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        # These will be calculated but not used in prediction
        "balanceDiffOrig": oldbalanceOrg - newbalanceOrig,
        "balanceDiffDest": newbalanceDest - oldbalanceDest
    }])
    
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_prob = model.predict_proba(input_data)[0][1]
        
        # Display results
        st.subheader("Prediction Results")
        st.write(f"Prediction: {'Fraudulent' if prediction == 1 else 'Legitimate'}")
        st.write(f"Fraud Probability: {prediction_prob:.2%}")
        
        if prediction == 1:
            st.error("⚠️ Warning: This transaction is predicted to be fraudulent")
            st.markdown("""
            **Potential fraud indicators:**
            - Large amount transferred out
            - Sender account emptied
            - Transaction type is TRANSFER or CASH_OUT
            """)
        else:
            st.success("✅ This transaction appears legitimate")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
