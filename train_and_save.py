import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import joblib

try:
    # --- 1. Load your final dataset ---
    df = pd.read_csv('merged_data.csv')
    print("--- Successfully loaded merged_data.csv ---")

    # --- 2. Define Target and Features ---
    df['target'] = (df['Riots'] > 0).astype(int)

    # Use the final, corrected list of features for the model
    features = [
        'Total Population', 'Piped (tap) water inside dwelling',
        'No access to piped (tap) water', 'Formal Dwelling',
        'Informal Dwelling', 'Flush toilet', 'No toilet',
        'Removed by local authority/private company/community members at least once a week',
        'Dump or leave rubbish anywhere (no rubbish disposal)',
        'Electricity for Light', 'Candles for Light',
        'Electricity for Cooking', 'Wood for Cooking',
        'Province name'
    ]

    # Clean the Total Population column if it's a string
    if 'Total Population' in df.columns and df['Total Population'].dtype == 'object':
        df['Total Population'] = df['Total Population'].str.replace(',', '').astype(float)
    
    X = df[features]
    y = df['target']
    
    print(f"Training model with {len(features)} features.")

    # --- 3. Define the Preprocessing Pipeline ---
    numerical_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

    # --- 4. Create and Train the Final Model Pipeline ---
    final_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', final_model)
    ])

    full_pipeline.fit(X, y)
    print("\n--- Final model has been trained successfully! ---")

    # --- 5. Save the Pipeline to a File ---
    model_filename = 'protest_risk_model.joblib'
    joblib.dump(full_pipeline, model_filename)
    
    print(f"Model pipeline saved to '{model_filename}'")

except FileNotFoundError:
    print("Error: 'merged_data.csv' not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
