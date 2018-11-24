import os
import sys


def makedirs(path, raise_error=True):
    try:
        os.makedirs(path)
    except Exception as e:
        if raise_error:
            raise e
        else:
            sys.exit(1)
