"""Write complete result records atomically before continuing an experiment."""

import json
import os


def save_json(path, result):
    """Replace a complete JSON record atomically and flush it to disk."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    os.fsync(directory)
    os.close(directory)
