import subprocess
import sys
from importlib import import_module

try:
    from loguru import logger
except ImportError:
    # Fallback if loguru isn't installed
    class _Logger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARN] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
    logger = _Logger()

# Last Cleanup: 2026-01-04

def update_emerge(branch: str = "main", confirm: bool = True, _dryrun: bool = False) -> str | None:
    """
    Update the EMerge library directly from GitHub.

    Parameters
    ----------
    branch : str, optional
        The git branch or tag to install from. Default is "main".
    confirm : bool, optional
        If True, asks the user for confirmation before updating.

    Returns
    -------
    str | None
        The updated version string if successful, otherwise None.
    """
    logger.warning(
        "You are about to update EMerge from GitHub. "
        "This is an experimental feature and may overwrite your installed version."
    )

    if confirm:
        ans = input("Do you wish to proceed? [y/N] ").strip().lower()
        if ans != "y":
            logger.info("Update aborted by user.")
            return None

    url = f"git+https://github.com/FennisRobert/EMerge.git@{branch}"
    logger.info(f"Updating EMerge from branch/tag '{branch}'...")

    if not _dryrun:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", url]
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Update failed: {e}")
            return None
    else:
        logger.info('Dry run... No update executed!')
    try:
        emerge = import_module("emerge")
        version = getattr(emerge, "__version__", "unknown")
        logger.info(f"Successfully updated to EMerge version {version}.")
        return version
    except Exception as e:
        logger.error(f"Update installed but could not retrieve version: {e}")
        return None