from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from sar_pattern_validation.workflows import (
    WorkflowResultCLIExcludedFields,
    complete_workflow,
)


def _serialize(obj: Any) -> Any:
    """
    Recursively convert objects into JSON-serializable structures.
    Handles:
      - dataclasses (excluding fields defined in WorkflowResultCLIExcludedFields)
      - pathlib.Path
      - nested dicts/lists
    """
    if is_dataclass(obj):
        excluded_fields = {field.value for field in WorkflowResultCLIExcludedFields}
        return {
            k: _serialize(v) for k, v in asdict(obj).items() if k not in excluded_fields
        }
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


def _configure_logging() -> None:
    """
    Ensure logs go to stderr (so stdout stays clean JSON).
    """
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint for sar-pattern-validation.

    - Parses CLI args (delegated to complete_workflow)
    - Runs workflow
    - Emits JSON result to stdout
    - Returns proper exit code
    """
    _configure_logging()

    try:
        # Pass argv through to your existing argparse logic
        args = argv if argv is not None else sys.argv[1:]

        result = complete_workflow(*args)

        # Convert result into JSON-safe structure
        payload = {
            "status": "success",
            "result": _serialize(result),
        }

        print(json.dumps(payload, indent=2))
        return 0

    except Exception as exc:
        from dataclasses import asdict

        from sar_pattern_validation.errors import WorkflowExecutionError

        issue_data = None
        if isinstance(exc, WorkflowExecutionError) and exc.issue is not None:
            issue_data = asdict(exc.issue)
        error_payload = {
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                **({"issue": issue_data} if issue_data else {}),
            },
        }

        # Print error JSON to stdout (not stderr) so frontend can always parse stdout
        # Logs remain on stderr for debugging
        print(json.dumps(error_payload, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
