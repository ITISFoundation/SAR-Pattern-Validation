from __future__ import annotations

from dataclasses import dataclass


class SarPatternValidationError(Exception):
    """Base exception for the SAR Pattern Validation package."""


class CsvFormatError(SarPatternValidationError):
    """Raised when a SAR CSV cannot be parsed or lacks required fields."""


class ConfigValidationError(SarPatternValidationError):
    """Raised when workflow configuration is invalid."""


class WorkflowExecutionError(SarPatternValidationError):
    """Raised when workflow execution fails."""

    def __init__(self, message: str, issue: ValidationIssue | None = None) -> None:
        super().__init__(message)
        self.issue = issue


@dataclass
class ValidationIssue:
    """A structured user-facing diagnostic emitted during workflow execution."""

    severity: str  # "warning" | "error"
    code: str
    message: str
    details: str | None = None
