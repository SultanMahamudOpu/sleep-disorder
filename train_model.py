import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import zscore

def train_and_save_model():
    # Load data
    df = pd.read_csv('sleep.csv')
    
    # Drop Person ID
    df.drop(columns='Person ID', inplace=True, axis=1)
    
    # Encode Gender
    level = LabelEncoder()
    df['Gender'] = level.fit_transform(df['Gender'])
    
    # Remove outliers using Z-score
    numeric_features = df.select_dtypes(include=['int64', 'float64'])
    z_scores = numeric_features.apply(zscore)
    outliers = (np.abs(z_scores) > 3)
    rows_without_outliers = ~(outliers.any(axis=1))
    df_no_outliers = df[rows_without_outliers].reset_index(drop=True)
    
    # Prepare features and target
    X = df_no_outliers.drop('Sleep Disorder', axis=1)
    y = df_no_outliers['Sleep Disorder']
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train Gradient Boosting Classifier
    param_grid_gb = {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid_search_gb = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_grid=param_grid_gb,
        scoring='f1_weighted',
        cv=5,
        n_jobs=1,
        verbose=1
    )
    
    grid_search_gb.fit(x_train, y_train)
    best_gb = grid_search_gb.best_estimator_
    
    # Evaluate model
    y_pred_gb = best_gb.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_gb)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Save model and encoders
    with open('sleep_disorder_model.pkl', 'wb') as f:
        pickle.dump(best_gb, f)
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open('gender_encoder.pkl', 'wb') as f:
        pickle.dump(level, f)
    
    # Save feature columns for reference
    feature_columns = X.columns.tolist()
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print("Model and encoders saved successfully!")
    print(f"Feature columns: {feature_columns}")
    print(f"Classes: {label_encoder.classes_}")
    
    return best_gb, label_encoder, level, feature_columns

if __name__ == "__main__":
    train_and_save_model()
