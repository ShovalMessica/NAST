import logging
import os
from datetime import datetime

def get_logger(name, log_dir=None, log_file=None, log_level=logging.INFO):
    """
    Create a logger instance.

    Args:
        name (str): Name of the logger.
        log_dir (str, optional): Directory to store the log file.
        log_file (str, optional): Name of the log file.
        log_level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create log directory if it doesn't exist
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a file handler
    if log_file:
        if log_dir:
            log_file_path = os.path.join(log_dir, log_file)
        else:
            log_file_path = log_file

        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
