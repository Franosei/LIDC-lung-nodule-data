"""Analysis package for the LIDC-IDRI misclassification study."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lidc-misclassification")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
