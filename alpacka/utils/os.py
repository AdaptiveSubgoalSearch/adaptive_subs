"""Operating system utilities."""

import contextlib
import os
import shutil


@contextlib.contextmanager
def _backup(path):
    """Makes a backup of a file/directory and restores it in case of error."""
    # The backup is saved in a hidden file within the same directory.
    (dir_path, file_name) = os.path.split(path)
    backup_path = os.path.join(dir_path, f'.{file_name}.backup')

    def remove(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)

    # Remove any old backup.
    remove(backup_path)
    # Move whathever is at path to the backup.
    os.replace(path, backup_path)

    try:
        # Do something with path.
        yield path

        # If everything goes well, remove the backup.
        remove(backup_path)
    except:
        # If not, remove the leftovers and restore the backup.
        remove(path)
        os.replace(backup_path, path)

        raise


@contextlib.contextmanager
def atomic_dump(paths):
    """Enables dumping data to several files or directories atomically.

    Useful for dumping big files. If the write is interrupted, no file is left
    in an invalid state. This is done by making backups of the old file contents
    and restoring them in case of an error.

    Note:
        Any of the paths need to be created again in the `with` block!

    Args:
        paths (tuple): Tuple of target paths.

    Yields:
        Tuple of paths to write to.
    """
    with contextlib.ExitStack() as stack:
        yield tuple(
            # Only backup files/directories that exist.
            stack.enter_context(_backup(path)) if os.path.exists(path) else path
            for path in paths
        )
