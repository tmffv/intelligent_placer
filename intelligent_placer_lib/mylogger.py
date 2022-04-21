import logging

_log_format = f"[%(asctime)s] - [%(levelname)s] - %(name)s - %(message)s"

def _get_file_handler():
    file_handler = logging.FileHandler("errors.log")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter(_log_format))
    return file_handler

def _get_stream_handler():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(_log_format))
    return stream_handler

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(_get_file_handler())
    logger.addHandler(_get_stream_handler())
    return logger
