"""Analysis package for the LIDC-IDRI annotation-effects study."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lidc-annotation-effects")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = ["__version__"]
