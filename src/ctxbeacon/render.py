"""Terminal rendering of compaction diffs, constraint findings, and ledger analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ctxbeacon.ledger import Ledger, LedgerError


def find_compaction_event(
    events: List[Dict[str, Any]],
    turn: Optional[int] = None,
    event_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Locate target compaction_completed event by event_id or turn number."""
    candidates = [e for e in events if e.get("event_type") == "compaction_completed"]
    if not candidates:
        return None

    if event_id:
        for c in candidates:
            if c.get("event_id") == event_id:
                return c
        return None

    if turn is not None:
        for c in candidates:
            if c.get("turn") == turn:
                return c
        return None

    # Default: most recent compaction
    return candidates[-1]


def find_constraint_findings(
    events: List[Dict[str, Any]],
    turn: Optional[int],
) -> List[Dict[str, Any]]:
    """Find constraint_check findings occurring at or directly after given turn."""
    findings: List[Dict[str, Any]] = []
    for e in events:
        if e.get("event_type") == "constraint_check":
            if turn is None or e.get("turn") == turn:
                findings.extend(e.get("constraint_findings", []))
    return findings


def render_compaction_diff(
    ledger_path: str | Path,
    turn: Optional[int] = None,
    event_id: Optional[str] = None,
    budget_override: Optional[int] = None,
    snapshots_dir: Optional[Path] = None,
) -> str:
    """
    Format and return a human-readable terminal diff for a context compaction event.
    """
    path = Path(ledger_path)
    if not path.exists():
        return f"Error: Ledger file not found at '{path}'."

    ledger = Ledger(path)
    try:
        events = ledger.read_all()
    except LedgerError as e:
        return f"Error: Failed reading ledger file: {e}"

    if not events:
        return f"Ledger '{path}' is empty. No events found."

    comp_event = find_compaction_event(events, turn=turn, event_id=event_id)
    if not comp_event:
        if turn is not None:
            return f"No compaction event found for turn {turn} in '{path}'."
        if event_id:
            return f"No compaction event found with event ID '{event_id}' in '{path}'."
        return f"No compaction events found in '{path}'."

    # Extract configured budget from run_started if available
    budget = budget_override
    if budget is None:
        for e in events:
            if e.get("event_type") == "run_started" and "budget_tokens" in e:
                budget = e.get("budget_tokens")
                break
    if not budget:
        budget = 4096

    event_turn = comp_event.get("turn", 0)
    c_before = comp_event.get("context_before_tokens", 0)
    c_after = comp_event.get("context_after_tokens", 0)
    pct_before = int((c_before / budget) * 100) if budget else 0
    pct_after = int((c_after / budget) * 100) if budget else 0

    retained_ids = comp_event.get("retained_ids", [])
    dropped_ids = comp_event.get("dropped_ids", [])
    summary_ref = comp_event.get("summary_ref")
    snapshot_ref = comp_event.get("snapshot_ref")
    comp_evt_id = comp_event.get("event_id", "unknown")

    # Inspect snapshot availability
    snap_status = "not enabled (original message content was not saved to disk; IDs and hashes recorded only)"
    snapshot_data: Optional[Dict[str, Any]] = None
    if snapshot_ref:
        base_dir = snapshots_dir or path.parent / "snapshots"
        snap_file = base_dir / f"{snapshot_ref}.json"
        if snap_file.exists():
            snap_status = f"available ({snap_file})"
            try:
                snapshot_data = json.loads(snap_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        else:
            snap_status = f"referenced ({snapshot_ref}) but file missing"

    # Determine summary text
    summary_text = comp_event.get("summary_preview")
    if snapshot_data and "summary" in snapshot_data:
        summary_text = snapshot_data["summary"]
    if not summary_text:
        summary_text = "(summary text not stored in event record)"

    # Check constraint findings
    findings = find_constraint_findings(events, event_turn)
    missing_findings = [f for f in findings if f.get("status") in ("possibly_missing", "possible_violation", "violation_observed")]
    preserved_findings = [f for f in findings if f.get("status") == "preserved"]

    # Build formatted terminal output
    lines: List[str] = []
    lines.append(f"TURN {event_turn}  CONTEXT {pct_before}% ({c_before}/{budget} tok) -> COMPACT {pct_after}% ({c_after}/{budget} tok)")
    lines.append("")
    lines.append("  RETAINED")
    if retained_ids:
        for mid in retained_ids:
            lines.append(f"    {mid}")
    else:
        lines.append("    (none)")

    lines.append("")
    lines.append("  SUMMARIZED / DROPPED")
    if dropped_ids:
        for mid in dropped_ids:
            lines.append(f"    {mid}")
    else:
        lines.append("    (none)")

    lines.append("")
    lines.append("  NEW SUMMARY")
    for s_line in summary_text.strip().split("\n"):
        lines.append(f"    {s_line}")

    if missing_findings:
        lines.append("")
        lines.append("  POSSIBLY MISSING")
        for mf in missing_findings:
            c_text = mf.get("constraint", "")
            lines.append(f'    "{c_text}"')
            lines.append("")
            lines.append("  EVIDENCE")
            lines.append(f"    Original message: {mf.get('evidence_message_id', 'unknown')}")
            lines.append(f"    Compaction event: {comp_evt_id}")
            kw_status = "absent" if not mf.get("matched_markers") else f"matched ({', '.join(mf['matched_markers'])})"
            lines.append(f"    Keyword match: {kw_status}")
            lines.append(f"    Snapshot: {snap_status}")
            if mf.get("details"):
                lines.append(f"    Details: {mf['details']}")
    elif preserved_findings:
        lines.append("")
        lines.append("  PRESERVED CONSTRAINTS")
        for pf in preserved_findings:
            lines.append(f'    "{pf.get("constraint", "")}" (keyword match: preserved)')
        lines.append("")
        lines.append("  EVIDENCE")
        lines.append(f"    Compaction event: {comp_evt_id}")
        lines.append(f"    Snapshot: {snap_status}")
    else:
        lines.append("")
        lines.append("  EVIDENCE")
        lines.append(f"    Compaction event: {comp_evt_id}")
        lines.append(f"    Snapshot: {snap_status}")

    lines.append("")
    lines.append("  Note: absence detected by deterministic keyword matching;")
    lines.append("        this does not prove that the model forgot the instruction.")
    return "\n".join(lines)
