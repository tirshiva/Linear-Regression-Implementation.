from logger import setup_logger
from utils import load_and_prepare_data, save_prediction_plot
from model import train_and_evaluate, save_model_and_scaler


def main():
    logger = setup_logger()
    logger.info("Starting Linear Regression pipeline.")

    try:
        X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(
            "house_data.csv", target_column="Price_lakhs"
        )
        model, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test, logger)
        save_prediction_plot(y_test, y_pred)
        save_model_and_scaler(model, scaler, version="v1")


        logger.info("All steps completed successfully.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")

if __name__ == "__main__":
    main()