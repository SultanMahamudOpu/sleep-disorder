import streamlit as st
import os

st.title("🔍 App Debug Info")

# Check if files exist
files_to_check = [
    'sleep_disorder_model.pkl',
    'label_encoder.pkl', 
    'gender_encoder.pkl',
    'occupation_encoder.pkl',
    'bmi_encoder.pkl',
    'feature_columns.pkl',
    'sleep.csv'
]

st.subheader("📁 File Status:")
for file in files_to_check:
    if os.path.exists(file):
        st.success(f"✅ {file} - Found")
    else:
        st.error(f"❌ {file} - Missing")

# Check current directory
st.subheader("📂 Current Directory:")
st.write(os.getcwd())

# List all files
st.subheader("📋 All Files in Directory:")
try:
    all_files = os.listdir('.')
    for file in sorted(all_files):
        st.write(f"- {file}")
except Exception as e:
    st.error(f"Error listing files: {e}")

# Test imports
st.subheader("📦 Package Versions:")
try:
    import pandas as pd
    st.write(f"Pandas: {pd.__version__}")
except:
    st.error("Pandas import failed")

try:
    import numpy as np
    st.write(f"NumPy: {np.__version__}")
except:
    st.error("NumPy import failed")

try:
    import sklearn
    st.write(f"Scikit-learn: {sklearn.__version__}")
except:
    st.error("Scikit-learn import failed")

try:
    import plotly
    st.write(f"Plotly: {plotly.__version__}")
except:
    st.error("Plotly import failed")
