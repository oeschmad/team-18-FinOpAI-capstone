import pandas as pd
import joblib
import os

def predict_risk(input_path='inputs/risk_input.csv',
                 model_path='models/risk_classifier.pkl',
                 encoder_path='models/risk_classifier_label_encoder.pkl'):
    """
    Predicts risk rating from user input CSV.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No input found at {input_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder file not found at {encoder_path}")

    # Load user input
    input_df = pd.read_csv(input_path)

    # Load trained model and label encoder
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)

    # Preprocessing: replicate training encoding
    payment_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}
    input_df['Payment History'] = input_df['Payment History'].map(payment_map)

    # One-hot encode categorical columns (match training structure)
    categorical = [
        'Gender', 'Education Level', 'Marital Status',
        'Loan Purpose', 'Employment Status',
        'City', 'State', 'Country', 'Marital Status Change'
    ]
    input_df = pd.get_dummies(input_df, columns=categorical, drop_first=True)

    # Align columns with model's expected features
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0  # Fill missing columns with 0
    input_df = input_df[model_features]  # Ensure correct order

    # Predict
    pred_encoded = model.predict(input_df)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    print(f"[INFO] Predicted Risk Rating: {pred_label}")

    return pred_label

if __name__ == "__main__":
    predict_risk()
