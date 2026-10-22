"""Deterministic constraint presence and violation checker for CtxBeacon."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

KEYWORD_LIMITATION_NOTE = (
    "Keyword matching cannot detect a paraphrased summary that preserves a constraint "
    "using different words. It is a deterministic diagnostic, not a semantic proof that "
    "the model remembers the requirement."
)


@dataclass
class ConstraintDefinition:
    """Specification of an explicit task requirement and audit markers."""
    text: str
    marker_terms: List[str] = field(default_factory=list)
    prohibited_paths: List[str] = field(default_factory=list)
    evidence_message_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintDefinition":
        return cls(
            text=data["text"],
            marker_terms=data.get("marker_terms", []),
            prohibited_paths=data.get("prohibited_paths", []),
            evidence_message_id=data.get("evidence_message_id"),
        )


@dataclass
class ConstraintFinding:
    """Outcome of a constraint presence or violation check."""
    constraint: str
    status: str  # "preserved", "possibly_missing", "possible_violation", "violation_observed", "unable_to_verify"
    marker_terms: List[str]
    matched_markers: List[str] = field(default_factory=list)
    evidence_message_id: Optional[str] = None
    evidence_tool_call_id: Optional[str] = None
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "constraint": self.constraint,
            "status": self.status,
            "marker_terms": self.marker_terms,
            "matched_markers": self.matched_markers,
            "details": self.details,
        }
        if self.evidence_message_id:
            d["evidence_message_id"] = self.evidence_message_id
        if self.evidence_tool_call_id:
            d["evidence_tool_call_id"] = self.evidence_tool_call_id
        return d


class ConstraintChecker:
    """
    Deterministic, auditable checker for constraint loss and violation detection.
    """

    @staticmethod
    def check_presence(
        constraint: ConstraintDefinition,
        context_messages: List[Dict[str, Any]],
    ) -> ConstraintFinding:
        """
        Check if any configured marker terms appear in the effective context.
        """
        if not constraint.marker_terms:
            return ConstraintFinding(
                constraint=constraint.text,
                status="unable_to_verify",
                marker_terms=[],
                evidence_message_id=constraint.evidence_message_id,
                details="No marker terms declared for constraint.",
            )

        matched_markers: List[str] = []
        # Filter for instructions and summaries, excluding raw tool calls and tool execution outputs
        relevant_messages = [
            m for m in context_messages
            if m.get("role") in ("system", "user") or (m.get("role") != "tool" and not m.get("tool_calls"))
        ]
        combined_text = " ".join(
            str(m.get("content", "")) for m in relevant_messages if m.get("content")
        ).lower()

        for marker in constraint.marker_terms:
            if marker.lower() in combined_text:
                matched_markers.append(marker)

        if matched_markers:
            return ConstraintFinding(
                constraint=constraint.text,
                status="preserved",
                marker_terms=constraint.marker_terms,
                matched_markers=matched_markers,
                evidence_message_id=constraint.evidence_message_id,
                details=(
                    f"Configured lexical check matched: {matched_markers}. "
                    f"{KEYWORD_LIMITATION_NOTE}"
                ),
            )

        return ConstraintFinding(
            constraint=constraint.text,
            status="possibly_missing",
            marker_terms=constraint.marker_terms,
            matched_markers=[],
            evidence_message_id=constraint.evidence_message_id,
            details=(
                "Original requirement was present before compaction but not found by "
                f"the post-compaction keyword check. {KEYWORD_LIMITATION_NOTE}"
            ),
        )

    @staticmethod
    def check_filesystem_violations(
        constraint: ConstraintDefinition,
        ledger_events: List[Dict[str, Any]],
        presence_finding: Optional[ConstraintFinding] = None,
    ) -> Optional[ConstraintFinding]:
        """
        Inspect executed tool calls to detect whether a prohibited write actually executed.
        
        Only reports 'violation_observed' when a prohibited write actually succeeded.
        A rejected or failed tool call is never described as a completed violation.
        """
        if not constraint.prohibited_paths:
            return None

        for evt in ledger_events:
            if evt.get("event_type") != "tool_call_completed":
                continue

            tool_name = evt.get("tool_name")
            status = evt.get("execution_status")
            target_path = evt.get("target_path") or ""

            # Check only file write tools
            if tool_name == "write_file":
                is_prohibited = any(
                    target_path == prob or target_path.startswith(prob.rstrip("/") + "/")
                    for prob in constraint.prohibited_paths
                )

                if is_prohibited:
                    if status == "success":
                        details = (
                            f"Violation observed: A tool call wrote to prohibited path '{target_path}'. "
                        )
                        if presence_finding and presence_finding.status == "possibly_missing":
                            details += (
                                "Possible cause: the original requirement was present before compaction "
                                "but not found by the post-compaction keyword check. "
                                "A later tool call wrote to the prohibited path."
                            )
                        return ConstraintFinding(
                            constraint=constraint.text,
                            status="violation_observed",
                            marker_terms=constraint.marker_terms,
                            matched_markers=presence_finding.matched_markers if presence_finding else [],
                            evidence_message_id=constraint.evidence_message_id,
                            evidence_tool_call_id=evt.get("tool_call_id"),
                            details=details,
                        )
                    elif status == "rejected":
                        # Attempt was blocked by sandbox, NOT a completed violation
                        return ConstraintFinding(
                            constraint=constraint.text,
                            status="possible_violation",
                            marker_terms=constraint.marker_terms,
                            matched_markers=presence_finding.matched_markers if presence_finding else [],
                            evidence_message_id=constraint.evidence_message_id,
                            evidence_tool_call_id=evt.get("tool_call_id"),
                            details=f"Write attempt to prohibited path '{target_path}' was blocked by sandbox.",
                        )

        return None

    @classmethod
    def audit_context_and_ledger(
        cls,
        constraints: List[ConstraintDefinition],
        context_messages: List[Dict[str, Any]],
        ledger_events: List[Dict[str, Any]],
    ) -> List[ConstraintFinding]:
        """
        Run complete presence and violation checks across configured constraints.
        """
        findings: List[ConstraintFinding] = []
        for c in constraints:
            pres_finding = cls.check_presence(c, context_messages)
            viol_finding = cls.check_filesystem_violations(c, ledger_events, presence_finding=pres_finding)
            
            # If a violation was observed, report that as priority finding
            if viol_finding:
                findings.append(viol_finding)
            else:
                findings.append(pres_finding)

        return findings
