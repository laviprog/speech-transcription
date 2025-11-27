import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

from fastapi import UploadFile

from src import log


def save_upload_file(upload_file: UploadFile, dir_save: str = "files") -> str:
    """
    Saves an uploaded file to the specified directory. Ensures the directory exists
    before saving the file. The file is saved with its original filename within the
    specified directory. If saving the file fails, an exception is raised. The path
    to the saved file is returned upon success.

    :param upload_file: The uploaded file to be saved.
    :param dir_save: Directory where the file should be saved, default is "files".
    :type dir_save: str
    :return: Path to the saved file as a string.
    :rtype: str
    """

    os.makedirs(dir_save, exist_ok=True)

    safe_filename = str(uuid.uuid4())

    path = Path(dir_save) / safe_filename

    try:
        upload_file.file.seek(0)
        with path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        log.debug("Upload file saved to %s", path)
    except Exception as e:
        log.error("Failed to save upload file: %s", e)
        raise

    return str(path)


def delete_file(path: str) -> None:
    """
    Deletes a file specified by the given path. This function checks if the file
    exists before attempting deletion and logs a warning if the file is not found.
    If an exception occurs during the deletion process, it is logged as an error.

    :param path: Path to the file to be deleted
    :type path: str
    :return: None
    """

    file_path = Path(path)

    if not file_path.exists():
        log.warning("File not found for deletion: %s", file_path)
        return

    try:
        file_path.unlink()
        log.debug("Deleted file: %s", file_path)
    except Exception as e:
        log.error("Failed to delete file %s: %s", file_path, e)


@contextmanager
def temporary_audio_file(file: UploadFile):
    """
    Context manager for handling temporary audio file operations. This context manager is designed
    to save an uploaded file temporarily to the server, yield its file path for any operations,
    and ensure the file is deleted after the operation has been completed, even in the case of
    exceptions.

    :param file: The uploaded file instance that needs to be processed and temporarily stored.
    :type file: UploadFile
    :yield: The file path of the temporarily saved audio file.
    :rtype: str
    """

    path = save_upload_file(file)
    try:
        yield path
    finally:
        delete_file(path)
