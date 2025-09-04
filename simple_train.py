import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import zscore

def train_and_save_model():
    # Load data
    df = pd.read_csv('sleep.csv')
    
    # Drop Person ID
    df.drop(columns='Person ID', inplace=True, axis=1)
    
    # Encode categorical features
    gender_encoder = LabelEncoder()
    df['Gender'] = gender_encoder.fit_transform(df['Gender'])
    
    occupation_encoder = LabelEncoder()
    df['Occupation'] = occupation_encoder.fit_transform(df['Occupation'])
    
    bmi_encoder = LabelEncoder()
    df['BMI Category'] = bmi_encoder.fit_transform(df['BMI Category'])
    
    # Handle Blood Pressure - extract systolic and diastolic
    bp_split = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic_BP'] = pd.to_numeric(bp_split[0])
    df['Diastolic_BP'] = pd.to_numeric(bp_split[1])
    df.drop('Blood Pressure', axis=1, inplace=True)
    
    # Remove outliers using Z-score
    numeric_features = df.select_dtypes(include=['int64', 'float64'])
    z_scores = numeric_features.apply(zscore)
    outliers = (np.abs(z_scores) > 3)
    rows_without_outliers = ~(outliers.any(axis=1))
    df_no_outliers = df[rows_without_outliers].reset_index(drop=True)
    
    print(f"Original data shape: {df.shape}")
    print(f"After removing outliers: {df_no_outliers.shape}")
    
    # Prepare features and target
    X = df_no_outliers.drop('Sleep Disorder', axis=1)
    y = df_no_outliers['Sleep Disorder']
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Classes: {label_encoder.classes_}")
    print(f"Feature columns: {X.columns.tolist()}")
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train Gradient Boosting Classifier with default parameters
    model = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42
    )
    
    print("Training model...")
    model.fit(x_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Save model and encoders
    with open('sleep_disorder_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open('gender_encoder.pkl', 'wb') as f:
        pickle.dump(gender_encoder, f)
    
    with open('occupation_encoder.pkl', 'wb') as f:
        pickle.dump(occupation_encoder, f)
    
    with open('bmi_encoder.pkl', 'wb') as f:
        pickle.dump(bmi_encoder, f)
    
    # Save feature columns for reference
    feature_columns = X.columns.tolist()
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print("Model and encoders saved successfully!")
    
    return model, label_encoder, gender_encoder, occupation_encoder, bmi_encoder, feature_columns

if __name__ == "__main__":
    train_and_save_model()
