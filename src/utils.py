"""
Utility functions for RAG Document Provenance Recovery.
"""

import os
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable


def setup_logging(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging for a module.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(level)

        # Detailed formatter for file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    # Simpler formatter for console
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    return logger


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    Execute function with exponential backoff retry.

    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Result of function execution

    Raises:
        Exception: If all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func()
        except exceptions as e:
            if attempt < max_retries - 1:
                wait_time = backoff ** attempt
                logging.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed after {max_retries} attempts: {e}")
                raise


@contextmanager
def timer(name: str = "Operation"):
    """
    Context manager to time a code block.

    Args:
        name: Name of the operation being timed

    Usage:
        with timer("Data collection"):
            collect_documents()
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[TIMER] {name} took {elapsed:.2f}s")


def inspect_data(data: Any, name: str = "Data", max_items: int = 3):
    """
    Print detailed information about a data structure.

    Args:
        data: Data structure to inspect
        name: Name of the data
        max_items: Maximum number of items to show
    """
    print(f"\n{'='*60}")
    print(f"INSPECTING: {name}")
    print(f"{'='*60}")
    print(f"Type: {type(data)}")

    if isinstance(data, list):
        print(f"Length: {len(data)}")
        if len(data) > 0:
            print(f"First item type: {type(data[0])}")
            for i, item in enumerate(data[:max_items]):
                print(f"[{i}]: {str(item)[:200]}")
    elif isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for i, (k, v) in enumerate(list(data.items())[:max_items]):
            print(f"  {k}: {type(v)} = {str(v)[:100]}")
    else:
        print(f"Value: {str(data)[:200]}")

    print(f"{'='*60}\n")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in text for a given model.

    Args:
        text: Text to count tokens for
        model: Model name

    Returns:
        int: Number of tokens
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logging.warning(f"Could not count tokens: {e}")
        # Rough estimate: 1 token ~= 4 characters
        return len(text) // 4


def validate_output(func: Callable, validation_func: Callable) -> Any:
    """
    Execute function and validate its output.

    Args:
        func: Function to execute
        validation_func: Function that returns True if output is valid

    Returns:
        Validated output

    Raises:
        ValueError: If validation fails
    """
    result = func()

    if not validation_func(result):
        raise ValueError(f"Output validation failed for {func.__name__}")

    return result
