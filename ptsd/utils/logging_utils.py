import os
import sys
import logging


class TeeStream:
    """
    Stream-like object that writes to multiple streams simultaneously.
    Used for redirecting stdout/stderr to both console and log files.
    """

    def __init__(self, streams):
        """
        Initialize with a list of output streams.

        Args:
            streams: List of file-like objects to write to
        """
        self.streams = streams

    def write(self, data):
        """Write data to all streams."""
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        """Flush all streams."""
        for stream in self.streams:
            if hasattr(stream, 'flush'):
                stream.flush()

    # For compatibility with file-like objects
    def isatty(self):
        """Return whether the first stream is a TTY."""
        return hasattr(self.streams[0], 'isatty') and self.streams[0].isatty()

    def fileno(self):
        """Return the file descriptor of the first stream."""
        return self.streams[0].fileno()


def setup_io_logging(run_dir):
    """
    Set up logging to redirect stdout and stderr to log files.

    Args:
        run_dir: Directory to save log files

    Returns:
        tuple: (original_stdout, original_stderr, run_log_file, error_log_file)
    """
    # Create file streams for logging
    run_log_file = open(os.path.join(run_dir, 'run.log'), 'a')
    error_log_file = open(os.path.join(run_dir, 'error.log'), 'a')

    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Create tee streams that write to both files and original streams
    sys.stdout = TeeStream([original_stdout, run_log_file])
    sys.stderr = TeeStream([original_stderr, error_log_file])

    return original_stdout, original_stderr, run_log_file, error_log_file


def setup_py_logging(
    original_stdout=None, original_stderr=None, log_level=logging.INFO
):
    """
    Configure Python logging module with handlers for stdout and stderr.

    Args:
        original_stdout: Original stdout stream (if None, uses sys.stdout)
        original_stderr: Original stderr stream (if None, uses sys.stderr)
        log_level: Logging level (default: logging.INFO)

    Returns:
        logger: Configured logger instance
    """
    # Use provided streams or current sys streams
    stdout_stream = original_stdout if original_stdout is not None else sys.stdout
    stderr_stream = original_stderr if original_stderr is not None else sys.stderr

    # Set up normal logging as well
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configure handlers
    stdout_handler = logging.StreamHandler(stdout_stream)
    stderr_handler = logging.StreamHandler(stderr_stream)

    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.ERROR)

    # Get root logger and configure
    logger = logging.getLogger()
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.setLevel(log_level)

    return logger


def setup_logging(run_dir, log_level=logging.INFO):
    """
    Set up both I/O redirection and Python logging.

    Args:
        run_dir: Directory to save log files
        log_level: Logging level (default: logging.INFO)

    Returns:
        logger: Configured logger instance
    """
    original_stdout, original_stderr, _, _ = setup_io_logging(run_dir)
    logger = setup_py_logging(original_stdout, original_stderr, log_level)
    return logger
