import logging

# This function sets up a logger that saves logs into training.log
def setup_logger():
    logger = logging.getLogger("LinearRegressionLogger")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("training.log", mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger