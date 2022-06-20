"""Compression utils."""

import joblib


def dump(value, path):
    """Dumps a Python value to a compressed pickle."""
    # Use joblib instead of pickle + gzip, because it's much more memory
    # efficient. The latter OOMs out for bigger replay buffers.
    joblib.dump(
        value,
        path,
        # Use protocol 4 - older protocols don't support big files.
        protocol=4,
        # Gzip compression at level 4 strikes a good balance between compression
        # speed and the size of the file.
        # Results of compressing a sample replay buffer:
        # level     size    time
        # 0         32GB    245s
        # 3         100MB   77s
        # 4         33MB    215s
        # 9         33MB    221s
        compress=('gzip', 4),
    )


def load(path):
    """Loads a Python value from a compressed pickle."""
    return joblib.load(path)
