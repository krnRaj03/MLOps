import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)

# Download the model 
model_path = hf_hub_download(
    repo_id="Hugo014/Tourism-Model",  # ← Changed from Tourism-Package-Prediction
    filename="best_tourism_model_v2.joblib",
    repo_type="model",  # ← Add this explicitly
    token=os.environ["HF_TOKEN"]  # ← Pass token
)
model = joblib.load(model_path)
print("Model loaded successfully!")

# Streamlit UI for Tourism Package Prediction
st.title("🌴 Wellness Tourism Package Prediction")
st.write("""
This application predicts whether a customer is likely to purchase the **Wellness Tourism Package** 
based on their demographic and interaction data. Enter the customer details below to get a prediction.
""")

st.sidebar.header("📋 Customer Information")

# Customer Demographics
st.sidebar.subheader("Demographics")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35, step=1)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
monthly_income = st.sidebar.number_input("Monthly Income ($)", min_value=0, max_value=100000, value=25000, step=1000)

# Family & Travel Details
st.sidebar.subheader("Travel Preferences")
number_of_person_visiting = st.sidebar.number_input("Number of People Traveling", min_value=1, max_value=10, value=2, step=1)
number_of_children_visiting = st.sidebar.number_input("Number of Children (Below 5)", min_value=0, max_value=5, value=0, step=1)
preferred_property_star = st.sidebar.slider("Preferred Hotel Star Rating", min_value=3.0, max_value=5.0, value=4.0, step=0.5)
number_of_trips = st.sidebar.number_input("Average Trips Per Year", min_value=0, max_value=20, value=2, step=1)

# Assets & Documentation
st.sidebar.subheader("Assets")
own_car = st.sidebar.selectbox("Own Car", ["No", "Yes"])
passport = st.sidebar.selectbox("Valid Passport", ["No", "Yes"])

# Professional Details
st.sidebar.subheader("Professional Info")
designation = st.sidebar.selectbox("Designation", ["Manager", "Senior Manager", "AVP", "VP", "Executive"])
city_tier = st.sidebar.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

# Interaction Data
st.sidebar.subheader("Sales Interaction")
type_of_contact = st.sidebar.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
product_pitched = st.sidebar.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
pitch_satisfaction_score = st.sidebar.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
number_of_followups = st.sidebar.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2, step=1)
duration_of_pitch = st.sidebar.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=15.0, step=1.0)

# Encode inputs to match training data
def encode_inputs():
    # Gender encoding (Female=0, Male=1)
    gender_encoded = 0 if gender == "Female" else 1
    
    # TypeofContact encoding (Company Invited=0, Self Inquiry=1)
    contact_encoded = 1 if type_of_contact == "Self Inquiry" else 0
    
    # Occupation encoding
    occupation_map = {"Salaried": 2, "Small Business": 0, "Large Business": 3, "Free Lancer": 1}
    occupation_encoded = occupation_map[occupation]
    
    # MaritalStatus encoding (Single=2, Married=0, Divorced=1, Unmarried=3)
    marital_map = {"Single": 2, "Married": 0, "Divorced": 1, "Unmarried": 3}
    marital_encoded = marital_map[marital_status]
    
    # ProductPitched ordinal encoding (Basic=0, Standard=1, Deluxe=2, Super Deluxe=3, King=4)
    product_map = {"Basic": 0, "Standard": 1, "Deluxe": 2, "Super Deluxe": 3, "King": 4}
    product_encoded = product_map[product_pitched]
    
    # Designation ordinal encoding (Manager=0, Senior Manager=1, AVP=2, VP=3, Executive=4)
    designation_map = {"Manager": 0, "Senior Manager": 1, "AVP": 2, "VP": 3, "Executive": 4}
    designation_encoded = designation_map[designation]
    
    # CityTier (extract number)
    city_tier_num = int(city_tier.split()[-1])
    
    # Binary encodings
    own_car_encoded = 1 if own_car == "Yes" else 0
    passport_encoded = 1 if passport == "Yes" else 0
    
    return {
        'Age': age,
        'TypeofContact': contact_encoded,
        'CityTier': city_tier_num,
        'Occupation': occupation_encoded,
        'Gender': gender_encoded,
        'NumberOfPersonVisiting': number_of_person_visiting,
        'PreferredPropertyStar': preferred_property_star,
        'MaritalStatus': marital_encoded,
        'NumberOfTrips': number_of_trips,
        'Passport': passport_encoded,
        'OwnCar': own_car_encoded,
        'NumberOfChildrenVisiting': number_of_children_visiting,
        'Designation': designation_encoded,
        'MonthlyIncome': monthly_income,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'ProductPitched': product_encoded,
        'NumberOfFollowups': number_of_followups,
        'DurationOfPitch': duration_of_pitch
    }

# Main content area
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔮 Predict Purchase Likelihood", use_container_width=True):
        # Prepare input data
        input_dict = encode_inputs()
        input_data = pd.DataFrame([input_dict])
        
        # Make prediction
        with st.spinner("Analyzing customer profile..."):
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        if prediction == 1:
            st.success("✅ **High Purchase Likelihood**")
            st.write(f"This customer is **likely to purchase** the Wellness Tourism Package.")
            st.metric("Confidence", f"{prediction_proba[1]*100:.1f}%")
            st.info("💡 **Recommendation**: Prioritize this lead for sales outreach!")
        else:
            st.warning("❌ **Low Purchase Likelihood**")
            st.write(f"This customer is **unlikely to purchase** the package at this time.")
            st.metric("Confidence", f"{prediction_proba[0]*100:.1f}%")
            st.info("💡 **Recommendation**: Consider nurturing this lead or offering alternative packages.")
        
        # Show probability breakdown
        st.markdown("---")
        st.subheader("📈 Probability Breakdown")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Will NOT Purchase", f"{prediction_proba[0]*100:.1f}%")
        with col_b:
            st.metric("Will Purchase", f"{prediction_proba[1]*100:.1f}%")

# Footer
st.markdown("---")
st.caption("🏨 Visit with Us - Tourism Package Prediction System | Powered by XGBoost ML Model")
