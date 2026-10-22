"""Append-only JSONL ledger reader, writer, and validator for CtxBeacon."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ctxbeacon.schema import ValidationError, validate_event


class LedgerError(Exception):
    """Base exception for ledger operations."""
    pass


class LedgerParseError(LedgerError):
    """Raised when parsing or validating an existing ledger record fails."""
    def __init__(self, message: str, line_num: int):
        super().__init__(f"Line {line_num}: {message}")
        self.line_num = line_num


class Ledger:
    """
    Append-only ledger backed by a JSONL file.
    
    Every record is validated against the CtxBeacon schema before appending,
    and written with immediate flushing to prevent data loss.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: Dict[str, Any]) -> None:
        """
        Validate and append an event record to the ledger.
        
        Preserves all earlier records and guarantees one JSON object per line.
        """
        # Validate schema before touching the file
        validate_event(event)

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                line = json.dumps(event, sort_keys=True, ensure_ascii=False)
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            raise LedgerError(f"Failed writing to ledger at '{self.path}': {e}") from e

    def read_all(self, validate_schema: bool = True) -> List[Dict[str, Any]]:
        """
        Read all events from the ledger.
        
        Raises LedgerParseError with line number if any line is malformed or invalid.
        """
        if not self.path.exists():
            return []

        events: List[Dict[str, Any]] = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        data = json.loads(stripped)
                    except json.JSONDecodeError as e:
                        raise LedgerParseError(f"Malformed JSON: {e}", line_num=line_num) from e

                    if validate_schema:
                        try:
                            validate_event(data, line_num=line_num)
                        except ValidationError as e:
                            raise LedgerParseError(str(e), line_num=line_num) from e

                    events.append(data)
        except OSError as e:
            raise LedgerError(f"Failed reading ledger at '{self.path}': {e}") from e

        return events

    @classmethod
    def validate_file(cls, path: str | Path) -> Tuple[int, List[str]]:
        """
        Scan a ledger file and return (valid_count, errors).
        Does not raise on line errors; accumulates them for reporting.
        """
        file_path = Path(path)
        if not file_path.exists():
            return 0, [f"Ledger file does not exist: {file_path}"]

        valid_count = 0
        errors: List[str] = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        data = json.loads(stripped)
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: Malformed JSON ({e})")
                        continue

                    try:
                        validate_event(data, line_num=line_num)
                        valid_count += 1
                    except ValidationError as e:
                        errors.append(str(e))
        except OSError as e:
            errors.append(f"Filesystem error accessing '{file_path}': {e}")

        return valid_count, errors
