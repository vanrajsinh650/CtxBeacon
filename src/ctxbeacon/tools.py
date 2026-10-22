"""Restricted sandboxed tools for CtxBeacon local agent execution."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class SandboxSecurityError(PermissionError):
    """Raised when a tool action attempts to escape the configured workspace sandbox."""
    pass


@dataclass
class ToolExecutionResult:
    """Standard outcome of a sandboxed tool call."""
    status: str  # "success", "error", "rejected"
    output: str
    target_path: Optional[str] = None
    error: Optional[str] = None


class WorkspaceSandbox:
    """
    Sandboxed filesystem environment strictly confined to a single directory.
    
    Protects against:
    - Path traversal (../)
    - Symlinks resolving outside sandbox root
    - Access to home directory, ssh keys, env secrets, or system files
    """

    def __init__(self, root_dir: Union[str, Path]) -> None:
        self.root = Path(root_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve_safe_path(self, user_path: str) -> Path:
        """
        Resolve a relative or absolute user path safely within sandbox root.
        Raises SandboxSecurityError if path escapes the sandbox.
        """
        raw_str = str(user_path).strip()
        if not raw_str:
            raise SandboxSecurityError("Empty path is not permitted")

        # Strip leading slashes to prevent arbitrary filesystem root access
        clean_rel = raw_str.lstrip("/")
        target = (self.root / clean_rel).resolve()

        # Strict containment check
        try:
            target.relative_to(self.root)
        except ValueError:
            raise SandboxSecurityError(f"Security boundary violation: '{user_path}' escapes workspace sandbox '{self.root}'")

        # Also verify if target exists and is a symlink resolving outside root
        if target.is_symlink():
            real_target = target.resolve()
            try:
                real_target.relative_to(self.root)
            except ValueError:
                raise SandboxSecurityError(f"Symlink traversal violation: '{user_path}' resolves outside sandbox")

        return target

    def list_files(self, subdir: str = ".") -> str:
        """List all regular files recursively or under a safe subdirectory."""
        target_dir = self.resolve_safe_path(subdir)
        if not target_dir.exists():
            return f"Directory not found: {subdir}"
        if not target_dir.is_dir():
            return f"Not a directory: {subdir}"

        entries: List[str] = []
        for path in sorted(target_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(self.root)
                entries.append(str(rel))

        if not entries:
            return "(empty directory)"
        return "\n".join(entries)

    def read_file(self, path: str) -> str:
        """Read content of a file within sandbox."""
        safe_path = self.resolve_safe_path(path)
        if not safe_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if safe_path.is_dir():
            raise IsADirectoryError(f"Target is a directory, not a file: {path}")

        return safe_path.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> str:
        """Write text content to a file within sandbox."""
        safe_path = self.resolve_safe_path(path)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
        rel_str = str(safe_path.relative_to(self.root))
        return f"Wrote {len(content)} characters to {rel_str}"

    def execute_tool(self, name: str, arguments: Union[Dict[str, Any], str]) -> ToolExecutionResult:
        """
        Execute tool safely with argument validation and error trapping.
        Returns ToolExecutionResult without raising unhandled exceptions.
        """
        # Parse arguments if passed as JSON string
        args: Dict[str, Any] = {}
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except Exception as e:
                return ToolExecutionResult(
                    status="error",
                    output=f"Invalid JSON arguments: {e}",
                    error=str(e),
                )
        elif isinstance(arguments, dict):
            args = arguments
        else:
            return ToolExecutionResult(
                status="error",
                output=f"Arguments must be dict or JSON string, got {type(arguments).__name__}",
                error="Invalid arguments type",
            )

        try:
            if name == "list_files":
                subdir = args.get("subdir", ".")
                out = self.list_files(subdir)
                return ToolExecutionResult(status="success", output=out)

            elif name == "read_file":
                if "path" not in args:
                    return ToolExecutionResult(status="error", output="Missing required argument 'path'", error="missing_argument")
                file_path = args["path"]
                out = self.read_file(file_path)
                return ToolExecutionResult(status="success", output=out, target_path=file_path)

            elif name == "write_file":
                if "path" not in args:
                    return ToolExecutionResult(status="error", output="Missing required argument 'path'", error="missing_argument")
                if "content" not in args:
                    return ToolExecutionResult(status="error", output="Missing required argument 'content'", error="missing_argument")
                file_path = args["path"]
                content = args["content"]
                out = self.write_file(file_path, content)
                return ToolExecutionResult(status="success", output=out, target_path=file_path)

            else:
                return ToolExecutionResult(
                    status="error",
                    output=f"Unknown tool '{name}'. Supported tools: list_files, read_file, write_file",
                    error="unknown_tool",
                )

        except SandboxSecurityError as e:
            return ToolExecutionResult(
                status="rejected",
                output=f"Access blocked by security sandbox: {e}",
                target_path=args.get("path"),
                error=str(e),
            )
        except Exception as e:
            return ToolExecutionResult(
                status="error",
                output=f"Tool execution failed: {e}",
                target_path=args.get("path"),
                error=str(e),
            )


# Tool schemas exposed to model providers
DEMO_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files within the demo workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subdir": {
                        "type": "string",
                        "description": "Subdirectory to list (default: workspace root)",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file inside the workspace sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path inside the workspace.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite text content to a file inside the workspace sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path inside the workspace.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]
