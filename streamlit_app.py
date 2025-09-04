import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Sleep Disorder Prediction",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1e88e5;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and encoders
@st.cache_resource
def load_model_and_encoders():
    try:
        with open('sleep_disorder_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open('gender_encoder.pkl', 'rb') as f:
            gender_encoder = pickle.load(f)
        
        with open('occupation_encoder.pkl', 'rb') as f:
            occupation_encoder = pickle.load(f)
        
        with open('bmi_encoder.pkl', 'rb') as f:
            bmi_encoder = pickle.load(f)
        
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        return model, label_encoder, gender_encoder, occupation_encoder, bmi_encoder, feature_columns
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None, None

# Load model and data
model, label_encoder, gender_encoder, occupation_encoder, bmi_encoder, feature_columns = load_model_and_encoders()

# Load original data for visualization
@st.cache_data
def load_data():
    return pd.read_csv('sleep.csv')

df = load_data()

# Main title
st.markdown('<h1 class="main-header">😴 Sleep Disorder Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict sleep disorders using advanced machine learning</p>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Data Visualization", "About"])

if page == "Prediction":
    st.header("🔮 Make a Prediction")
    
    if model is not None:
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", min_value=18, max_value=80, value=30)
            occupation = st.selectbox("Occupation", [
                "Software Engineer", "Doctor", "Sales Representative", "Teacher", 
                "Nurse", "Engineer", "Accountant", "Scientist", "Lawyer", 
                "Salesperson", "Manager"
            ])
            
        with col2:
            st.subheader("Health Metrics")
            sleep_duration = st.slider("Sleep Duration (hours)", min_value=4.0, max_value=10.0, value=7.0, step=0.1)
            quality_of_sleep = st.slider("Quality of Sleep (1-10)", min_value=1, max_value=10, value=7)
            physical_activity = st.slider("Physical Activity Level (0-100)", min_value=0, max_value=100, value=50)
            stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Physical Characteristics")
            bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
            heart_rate = st.slider("Heart Rate (bpm)", min_value=50, max_value=120, value=70)
            
        with col4:
            st.subheader("Vital Signs")
            systolic_bp = st.slider("Systolic Blood Pressure", min_value=90, max_value=180, value=120)
            diastolic_bp = st.slider("Diastolic Blood Pressure", min_value=60, max_value=120, value=80)
            daily_steps = st.slider("Daily Steps", min_value=1000, max_value=15000, value=7000)
        
        # Prediction button
        if st.button("🔍 Predict Sleep Disorder", type="primary"):
            try:
                # Prepare input data
                input_data = pd.DataFrame({
                    'Gender': [gender_encoder.transform([gender])[0]],
                    'Age': [age],
                    'Occupation': [occupation_encoder.transform([occupation])[0]],
                    'Sleep Duration': [sleep_duration],
                    'Quality of Sleep': [quality_of_sleep],
                    'Physical Activity Level': [physical_activity],
                    'Stress Level': [stress_level],
                    'BMI Category': [bmi_encoder.transform([bmi_category])[0]],
                    'Heart Rate': [heart_rate],
                    'Daily Steps': [daily_steps],
                    'Systolic_BP': [systolic_bp],
                    'Diastolic_BP': [diastolic_bp]
                })
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
                
                # Get the class names
                class_names = label_encoder.classes_
                predicted_class = class_names[prediction]
                
                # Handle nan values in prediction
                if pd.isna(predicted_class):
                    predicted_class = "No Sleep Disorder"
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Prediction Result</h2>
                    <h1>{predicted_class}</h1>
                    <p>Confidence: {max(prediction_proba):.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show probability distribution
                st.subheader("Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Condition': [c if not pd.isna(c) else "No Sleep Disorder" for c in class_names],
                    'Probability': prediction_proba
                })
                
                fig = px.bar(prob_df, x='Condition', y='Probability', 
                           title="Probability Distribution",
                           color='Probability',
                           color_continuous_scale='viridis')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations based on prediction
                st.subheader("💡 Recommendations")
                if predicted_class == "Insomnia":
                    st.warning("""
                    **Insomnia detected.** Consider these recommendations:
                    - Maintain a regular sleep schedule
                    - Create a comfortable sleeping environment
                    - Limit caffeine and electronic devices before bedtime
                    - Practice relaxation techniques
                    - Consult a healthcare professional if symptoms persist
                    """)
                elif predicted_class == "Sleep Apnea":
                    st.error("""
                    **Sleep Apnea detected.** Important recommendations:
                    - Consult a healthcare professional immediately
                    - Consider sleep study evaluation
                    - Maintain a healthy weight
                    - Sleep on your side
                    - Avoid alcohol and sedatives before bedtime
                    """)
                else:
                    st.success("""
                    **No sleep disorder detected.** Keep up the good work:
                    - Continue maintaining healthy sleep habits
                    - Regular exercise and balanced diet
                    - Manage stress levels
                    - Monitor your sleep quality regularly
                    """)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    else:
        st.error("Model not loaded. Please check if the model files exist.")

elif page == "Data Visualization":
    st.header("📊 Data Insights")
    
    # Dataset overview
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns)-1)
    with col3:
        st.metric("Sleep Disorders", df['Sleep Disorder'].notna().sum())
    with col4:
        st.metric("Healthy Records", df['Sleep Disorder'].isna().sum())
    
    # Sleep disorder distribution
    st.subheader("Sleep Disorder Distribution")
    disorder_counts = df['Sleep Disorder'].fillna('No Disorder').value_counts()
    
    fig = px.pie(values=disorder_counts.values, names=disorder_counts.index,
                title="Distribution of Sleep Disorders")
    st.plotly_chart(fig, use_container_width=True)
    
    # Age vs Sleep Quality
    st.subheader("Age vs Sleep Quality Analysis")
    fig = px.scatter(df, x='Age', y='Quality of Sleep', 
                    color='Sleep Disorder',
                    title="Age vs Sleep Quality by Sleep Disorder",
                    hover_data=['Gender', 'Occupation'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Sleep Duration Distribution
    st.subheader("Sleep Duration Analysis")
    fig = px.histogram(df, x='Sleep Duration', 
                      color='Sleep Disorder',
                      title="Sleep Duration Distribution by Disorder Type",
                      nbins=20)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                   title="Correlation Matrix of Numeric Features",
                   color_continuous_scale='RdBu')
    st.plotly_chart(fig, use_container_width=True)

elif page == "About":
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Purpose</h3>
    <p>This application uses machine learning to predict sleep disorders based on various health and lifestyle factors. 
    It's designed to help individuals understand their sleep health and seek appropriate medical attention when necessary.</p>
    </div>
    
    <div class="info-box">
    <h3>🤖 Model Information</h3>
    <p>The prediction model is built using a Gradient Boosting Classifier, trained on a comprehensive dataset of sleep health metrics. 
    The model achieves high accuracy in predicting three main categories:</p>
    <ul>
        <li><strong>No Sleep Disorder:</strong> Normal, healthy sleep patterns</li>
        <li><strong>Insomnia:</strong> Difficulty falling or staying asleep</li>
        <li><strong>Sleep Apnea:</strong> Breathing interruptions during sleep</li>
    </ul>
    </div>
    
    <div class="info-box">
    <h3>📊 Features Used</h3>
    <p>The model considers the following factors:</p>
    <ul>
        <li>Personal demographics (Age, Gender, Occupation)</li>
        <li>Sleep metrics (Duration, Quality)</li>
        <li>Physical health (BMI, Heart Rate, Blood Pressure)</li>
        <li>Lifestyle factors (Physical Activity, Stress Level, Daily Steps)</li>
    </ul>
    </div>
    
    <div class="info-box">
    <h3>⚠️ Important Disclaimer</h3>
    <p><strong>This tool is for educational and informational purposes only.</strong> 
    It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult with qualified healthcare professionals for any sleep-related concerns.</p>
    </div>
    
    <div class="info-box">
    <h3>👨‍💻 Technical Details</h3>
    <p>Built with:</p>
    <ul>
        <li>Python & Streamlit for the web interface</li>
        <li>Scikit-learn for machine learning</li>
        <li>Plotly for interactive visualizations</li>
        <li>Pandas & NumPy for data processing</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Built with ❤️ using Streamlit | Sleep Health Prediction System</p>",
    unsafe_allow_html=True
)
