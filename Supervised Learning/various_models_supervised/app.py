import streamlit as st  
import pandas as pd
import joblib
import numpy as np


# Page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Theme
st.markdown("""
<style>
    /* Hide only hamburger menu, not the entire header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dark theme background */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Sidebar dark theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* Force all text to white */
    .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp span, .stApp div, .stApp label, .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Sidebar text white */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        animation: fadeIn 0.5s;
    }
    .high-risk {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white !important;
    }
    .low-risk {
        background: linear-gradient(135deg, #059669 0%, #065f46 100%);
        color: white !important;
    }
    .info-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin: 10px 0;
        border-left: 4px solid #3b82f6;
    }
    .info-card h3, .info-card h4, .info-card p, .info-card ul, .info-card li {
        color: #ffffff !important;
    }
    .metric-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
    }
    .metric-container h2, .metric-container h4, .metric-container p {
        color: #ffffff !important;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    h1 {
        color: #ffffff !important;
        text-align: center;
        padding: 20px 0;
        font-weight: 700;
    }
    
    /* Tabs styling dark */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        color: #ffffff !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
    }
    
    /* Input fields dark */
    input, textarea, select {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #475569 !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #1e293b !important;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
        border: none;
        font-weight: bold;
        padding: 12px 24px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* Success/Warning/Error boxes dark */
    .stSuccess, .stWarning, .stError, .stInfo {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: #ffffff !important;
    }
    
    /* Footer text */
    div[style*='text-align: center'] {
        color: #94a3b8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load saved model, scaler, and expected columns
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# Header
st.title("🫀 Advanced Heart Disease Risk Predictor")
st.markdown("""
<div class='info-card'>
    <h3>Welcome to the AI-Powered Heart Health Assessment</h3>
    <p>This tool uses machine learning to assess your risk of heart disease based on clinical parameters. 
    Please provide accurate information for the best prediction.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    pass

# Create tabs for organized input
tab1, tab2, tab3 = st.tabs(["📝 Basic Information", "🩺 Clinical Measurements", "📈 Additional Tests"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Personal Details")
        age = st.slider("Age (years)", 18, 100, 40, help="Your current age")
        sex = st.selectbox("Biological Sex", ["M", "F"], help="Male or Female")
        
    with col2:
        st.markdown("### Symptoms")
        chest_pain = st.selectbox(
            "Chest Pain Type", 
            ["ATA", "NAP", "TA", "ASY"],
            help="ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina, ASY: Asymptomatic"
        )
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina", 
            ["Y", "N"],
            help="Do you experience chest pain during exercise?"
        )

with tab2:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Blood Pressure")
        resting_bp = st.number_input(
            "Resting BP (mm Hg)", 
            80, 200, 120,
            help="Normal range: 90-120 mm Hg"
        )
        
    with col2:
        st.markdown("### Cholesterol")
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)", 
            100, 600, 200,
            help="Optimal: <200 mg/dL"
        )
        
    with col3:
        st.markdown("### Blood Sugar")
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL", 
            [0, 1],
            help="0: No, 1: Yes (Diabetic indicator)"
        )

with tab3:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Heart Rate")
        max_hr = st.slider(
            "Maximum Heart Rate", 
            60, 220, 150,
            help="Achieved during exercise test"
        )
        
    with col2:
        st.markdown("### ECG Results")
        resting_ecg = st.selectbox(
            "Resting ECG", 
            ["Normal", "ST", "LVH"],
            help="ST: ST-T wave abnormality, LVH: Left ventricular hypertrophy"
        )
        
    with col3:
        st.markdown("### ST Depression")
        oldpeak = st.slider(
            "Oldpeak (ST Depression)", 
            0.0, 6.0, 1.0, 0.1,
            help="Depression induced by exercise relative to rest"
        )
        st_slope = st.selectbox(
            "ST Slope", 
            ["Up", "Flat", "Down"],
            help="Slope of peak exercise ST segment"
        )

# Display metrics summary
st.markdown("---")
st.subheader("📊 Your Health Metrics Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-container'>
        <h4>Age</h4>
        <h2>{age}</h2>
        <p>years</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-container'>
        <h4>Blood Pressure</h4>
        <h2>{resting_bp}</h2>
        <p>mm Hg</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-container'>
        <h4>Cholesterol</h4>
        <h2>{cholesterol}</h2>
        <p>mg/dL</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-container'>
        <h4>Max Heart Rate</h4>
        <h2>{max_hr}</h2>
        <p>bpm</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Predict button with custom styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 Predict Heart Disease Risk", use_container_width=True, type="primary")

# When Predict is clicked
if predict_button:
    with st.spinner("🔄 Analyzing your data..."):
        # Create a raw input dictionary
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }
        
        # Creating a new data frame 
        input_df = pd.DataFrame([raw_input])
        
        # Fill in missing cols with 0's
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Reorder columns
        input_df = input_df[expected_columns]

        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        
        # Get prediction probability if available
        try:
            proba = model.predict_proba(scaled_input)[0]
            confidence = max(proba) * 100
        except:
            confidence = None
        
        # Show result
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        if prediction == 1:
            st.markdown(f"""
            <div class='prediction-box high-risk'>
                ⚠️ HIGH RISK OF HEART DISEASE
            </div>
            """, unsafe_allow_html=True)
            
            if confidence:
                st.error(f"Confidence Level: {confidence:.1f}%")
            
            st.markdown("""
            <div class='info-card'>
                <h4>⚕️ Recommended Actions:</h4>
                <ul>
                    <li>Consult a cardiologist immediately</li>
                    <li>Schedule comprehensive cardiac tests</li>
                    <li>Monitor blood pressure and cholesterol regularly</li>
                    <li>Adopt heart-healthy lifestyle changes</li>
                    <li>Review medication with your doctor</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
            <div class='prediction-box low-risk'>
                ✅ LOW RISK OF HEART DISEASE
            </div>
            """, unsafe_allow_html=True)
            
            if confidence:
                st.success(f"Confidence Level: {confidence:.1f}%")
            
            st.markdown("""
            <div class='info-card'>
                <h4>💚 Maintain Your Heart Health:</h4>
                <ul>
                    <li>Continue regular health check-ups</li>
                    <li>Maintain a balanced diet</li>
                    <li>Exercise regularly (150 min/week)</li>
                    <li>Manage stress effectively</li>
                    <li>Avoid smoking and excessive alcohol</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='info-card'>
                <h4>📌 Risk Factors Detected</h4>
            </div>
            """, unsafe_allow_html=True)
            
            risk_factors = []
            if age > 55:
                risk_factors.append("• Advanced age")
            if resting_bp > 140:
                risk_factors.append("• High blood pressure")
            if cholesterol > 240:
                risk_factors.append("• High cholesterol")
            if fasting_bs == 1:
                risk_factors.append("• Elevated blood sugar")
            if exercise_angina == "Y":
                risk_factors.append("• Exercise-induced angina")
                
            if risk_factors:
                for factor in risk_factors:
                    st.warning(factor)
            else:
                st.success("• No major risk factors detected")
        
        with col2:
            st.markdown("""
            <div class='info-card'>
                <h4>💪 Positive Indicators</h4>
            </div>
            """, unsafe_allow_html=True)
            
            positive_factors = []
            if age <= 55:
                positive_factors.append("• Younger age group")
            if resting_bp <= 120:
                positive_factors.append("• Normal blood pressure")
            if cholesterol <= 200:
                positive_factors.append("• Healthy cholesterol level")
            if max_hr >= 140:
                positive_factors.append("• Good heart rate capacity")
                
            if positive_factors:
                for factor in positive_factors:
                    st.success(factor)
            else:
                st.info("• Focus on lifestyle improvements")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 20px;'>
    <p>Developed by Mayank | Powered by Machine Learning</p>
    <p style='font-size: 12px;'>⚠️ This tool is for educational and informational purposes only. 
    It should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
</div>
""", unsafe_allow_html=True)