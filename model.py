from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Trains a linear regression model and evaluates its performance
def train_and_evaluate(X_train, X_test, y_train, y_test, logger):
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Model training completed.")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Mean Squared Error (MSE): {mse}")
    logger.info(f"R² Score: {r2}")

    return model, y_pred

# Saves model using joblib into the artifacts folder
def save_model_and_scaler(model, scaler, version="v1"):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/model_{version}.pkl")
    joblib.dump(scaler, f"models/scaler_{version}.pkl")