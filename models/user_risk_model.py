# risk_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_risk_model(csv_path='/data/financial_risk_assessment.csv',
                     save_path='models/risk_classifier.pkl',
                     use_smote=True):
    # Step 1: Load data
    df = pd.read_csv(csv_path)

    # Step 2: Drop rows with missing target
    df = df.dropna(subset=['Risk Rating'])

    # Step 3: Fill missing numeric values with median
    numeric_cols = ['Income', 'Credit Score', 'Loan Amount', 'Years at Current Job',
                    'Debt-to-Income Ratio', 'Assets Value', 'Number of Dependents', 'Previous Defaults']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Step 4: Ordinal encode 'Payment History'
    payment_map = {
        'Poor': 0,
        'Fair': 1,
        'Good': 2,
        'Very Good': 3,
        'Excellent': 4
    }
    df['Payment History'] = df['Payment History'].map(payment_map)

    # Step 5: One-hot encode categorical columns
    df = pd.get_dummies(df, columns=[
        'Gender', 'Education Level', 'Marital Status',
        'Loan Purpose', 'Employment Status',
        'City', 'State', 'Country', 'Marital Status Change'
    ], drop_first=True)

    # Step 6: Features & target
    X = df.drop(columns=['Risk Rating'])
    y = df['Risk Rating']

    # Step 7: Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Step 8: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Step 9: SMOTE Oversampling (if specified)
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("[INFO] Applied SMOTE for class balancing.")

    # Step 10: Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 11: Evaluate
    y_pred = model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Step 12: Save model and label encoder
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    joblib.dump(le, save_path.replace('.pkl', '_label_encoder.pkl'))
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_risk_model()
