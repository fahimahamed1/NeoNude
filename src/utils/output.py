"""
Output suppression utilities.

Provides context managers and functions to suppress verbose output
from model loading and inference processes.
"""

import os
import sys
import logging
from contextlib import contextmanager, redirect_stdout, redirect_stderr


def configure_quiet_mode():
    """Configure libraries to suppress verbose output.
    
    Sets environment variables and logging levels to minimize
    output from transformers and diffusers.
    Note: tqdm progress bars are NOT disabled to show download progress.
    """
    # Suppress transformers output
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    
    # Suppress diffusers output
    os.environ['DIFFUSERS_VERBOSITY'] = 'error'
    
    # Note: TQDM_DISABLE is NOT set to allow download progress bars
    
    # Configure logging levels
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('diffusers').setLevel(logging.ERROR)


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr.
    
    Usage:
        with suppress_output():
            # Verbose code here
            pass
    """
    devnull = open(os.devnull, 'w')
    try:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield
    finally:
        devnull.close()


@contextmanager
def suppress_stdout():
    """Context manager to suppress only stdout.
    
    Useful when you want to keep stderr for error messages.
    """
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
