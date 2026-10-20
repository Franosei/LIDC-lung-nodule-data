"""Domain-specific exceptions with actionable failure messages."""


class LidcBiasError(RuntimeError):
    """Base exception for expected pipeline failures."""


class SourceDataError(LidcBiasError):
    """Raised when a source file is missing, malformed or internally inconsistent."""


class ValidationError(LidcBiasError):
    """Raised when a validation rule fails."""


class AnalysisError(LidcBiasError):
    """Raised when an analysis cannot be evaluated as specified."""
