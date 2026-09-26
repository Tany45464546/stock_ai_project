import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

def train_xgboost_predictor(data_path="historical_stock_data.csv"):
    print("--> Loading advanced stock dataset...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Expanded Feature Set
    features = ['SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 
                'BB_Upper', 'BB_Lower', 'ATR_14', 'Volume_Change', 'Daily_Return']
    
    X = df[features]
    y = df['Target']

    # Sequential Time-Series Train/Test Split (80/20)
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"--> Training XGBoost on {len(X_train)} samples, testing on {len(X_test)} samples...")

    # Train XGBoost Model
    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "="*50)
    print(f" XGBoost Model Accuracy Score: {accuracy * 100:.2f}%")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Price DOWN', 'Price UP']))

    # Feature Importance Breakdown
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("Feature Importance Ranking:")
    print(importance_df.to_string(index=False))

    # Append predictions to main dataframe
    df['ML_Predicted_Signal'] = model.predict(X)
    probabilities = model.predict_proba(X)
    df['ML_Confidence'] = np.max(probabilities, axis=1)

    output_filename = "stock_data_with_ml_predictions.csv"
    df.to_csv(output_filename)
    print(f"\n--> Updated dataset saved to '{output_filename}'.")
    return df

if __name__ == "__main__":
    train_xgboost_predictor()
