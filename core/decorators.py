import logging
from functools import wraps
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_System")


def log_execution(func):
    """Decorator to log function execution time and errors."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            logger.info(f"Starting {func.__name__}...")
            result = func(*args, **kwargs)
            logger.info(f"Finished {func.__name__} in {time.time() - start:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise e

    return wrapper
