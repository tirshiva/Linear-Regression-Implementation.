import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Loads and prepares dataset for training/testing
def load_and_prepare_data(csv_path, target_column):
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler

# Saves prediction vs actual plot to artifacts folder
def save_prediction_plot(y_test, y_pred):
    os.makedirs("artifacts", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
    plt.xlabel("Actual Price (Lakhs)")
    plt.ylabel("Predicted Price (Lakhs)")
    plt.title("Actual vs Predicted House Prices")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.savefig("artifacts/prediction_plot.png")