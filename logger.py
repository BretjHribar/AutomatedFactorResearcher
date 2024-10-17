import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger with file and console handlers."""
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(funcName)s(): %(message)s')

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Create a logger for the main application
main_logger = setup_logger('main_logger', 'logs/main.log')
