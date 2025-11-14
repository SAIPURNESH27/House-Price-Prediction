import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# --- Load Model and Scaler ---
# Use st.cache_resource to load only once and speed up the app
@st.cache_resource
def load_model():
    """Loads the saved model and scaler from disk."""
    try:
        model = joblib.load('ridge_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or scaler file not found. Please run the notebook to create them.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred loading the files: {e}")
        return None, None

model, scaler = load_model()

# --- Feature Names (in the correct order) ---
# This order MUST match the order your model was trained on
feature_names = [
    'crim', 'zn', 'indus', 'chas', 'nox', 'rm',
    'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat'
]

# --- Main App Interface ---
st.title("🏠 Boston House Price Predictor")
st.markdown("Enter the features of a house to predict its price.")

if model and scaler:
    
    # --- Input Form ---
    # Using a form bundles inputs; prediction only runs on submit.
    with st.form(key="prediction_form"):
        st.subheader("Enter House Features:")
        
        # Create columns for a cleaner layout
        col1, col2, col3 = st.columns(3)
        
        # Dictionary to store user inputs
        inputs = {}
        
        # Dynamically create number inputs
        with col1:
            inputs['crim'] = st.number_input("CRIM (Crime Rate)", min_value=0.0, value=0.03, format="%.4f")
            inputs['zn'] = st.number_input("ZN (Residential Land %)", min_value=0.0, value=0.0, format="%.1f")
            inputs['indus'] = st.number_input("INDUS (Non-Retail Acres %)", min_value=0.0, value=7.0, format="%.2f")
            inputs['chas'] = st.selectbox("CHAS (River Proximity)", [0, 1], help="1 if tracts bounds river, 0 otherwise")

        with col2:
            inputs['nox'] = st.number_input("NOX (Nitric Oxides)", min_value=0.0, value=0.4, format="%.3f")
            inputs['rm'] = st.number_input("RM (Avg Rooms)", min_value=1.0, value=6.5, format="%.2f")
            inputs['age'] = st.number_input("AGE (Owner-Occupied %)", min_value=0.0, max_value=100.0, value=78.0, format="%.1f")
            inputs['dis'] = st.number_input("DIS (Dist to Employment)", min_value=0.0, value=4.9, format="%.3f")

        with col3:
            inputs['rad'] = st.number_input("RAD (Highway Access Index)", min_value=0.0, value=2.0, format="%.0f")
            inputs['tax'] = st.number_input("TAX (Property Tax Rate)", min_value=0.0, value=242.0, format="%.0f")
            inputs['ptratio'] = st.number_input("PTRATIO (Pupil-Teacher Ratio)", min_value=0.0, value=17.0, format="%.1f")
            inputs['b'] = st.number_input("B (Black Proportion)", min_value=0.0, value=396.0, format="%.2f")
            inputs['lstat'] = st.number_input("LSTAT (% Lower Status)", min_value=0.0, value=4.0, format="%.2f")

        # Submit Button
        submit_button = st.form_submit_button(label="✨ Predict Price!")

    # --- Prediction Logic ---
    if submit_button:
        # 1. Collect inputs in the correct order
        input_list = [inputs[name] for name in feature_names]
        
        # 2. Convert to 2D numpy array
        input_array = np.array(input_list).reshape(1, -1)
        
        # 3. Scale the inputs
        input_scaled = scaler.transform(input_array)
        
        # 4. Make prediction
        # --- Animation 1: Spinner ---
        with st.spinner("🧠 Model is thinking..."):
            time.sleep(1) # Small delay for dramatic effect
            prediction = model.predict(input_scaled)
            predicted_price = prediction[0] # Get the single value

        # 5. Display Result
        # --- Animation 2: Balloons ---
        st.balloons()
        st.success("Prediction Complete!")
        
        # Display the result in a clean metric box
        st.metric(
            label="Predicted House Price",
            value=f"${predicted_price * 1000:,.2f}",
            help="Note: The model predicts in units of $1000s"
        )
        
        st.markdown(f"*(Based on the inputs, the model predicts a price of **${predicted_price:.2f}K**.)*")

else:
    st.warning("Please make sure 'ridge_model.joblib' and 'scaler.joblib' are in the same folder as app.py.")