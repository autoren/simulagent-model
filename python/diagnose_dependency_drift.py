from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from cross_track_evidence_audit import ROOT, read_json, sha256_file, valid_lock, write_json


AUDIT_PATH = ROOT / "outputs/cross-track-evidence-audit-through-v224/reproducibility-audit.json"
LEDGER_PATH = ROOT / "outputs/cross-track-evidence-audit-through-v224/experiment-ledger.json"
OUTPUT_PATH = ROOT / "outputs/cross-track-evidence-audit-through-v224/dependency-drift-provenance-addendum.json"
DOCUMENT_PATH = ROOT / "docs/dependency-drift-provenance-addendum-through-v224.md"


def _git(args: list[str]) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _blob_at(commit: str, path: str) -> bytes | None:
    try:
        return _git(["show", f"{commit}:{path}"])
    except subprocess.CalledProcessError:
        return None


def _classification(path: str) -> str:
    if path.startswith("docs/"):
        return "narrative_results_document"
    if "verify_and_freeze" in path:
        return "executable_outcome_verifier"
    return "executable_auditor"


def diagnose() -> dict[str, Any]:
    audit = read_json(AUDIT_PATH)
    ledger = read_json(LEDGER_PATH)
    experiment_by_lock = {row["outcome_lock"]["path"]: row for row in ledger}
    findings = []
    for drift in audit["drifted_non_sensitive_dependencies"]:
        path = drift["path"]
        expected = drift["expected_sha256"]
        commits = _git(["log", "--all", "--reflog", "--format=%H", "--", path]).decode().split()
        versions = []
        for commit in commits:
            body = _blob_at(commit, path)
            if body is None:
                continue
            versions.append(
                {
                    "commit": commit,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "size_bytes": len(body),
                    "matches_frozen_expected_sha256": hashlib.sha256(body).hexdigest() == expected,
                }
            )
        status = _git(["status", "--short", "--", path]).decode().strip()
        source = experiment_by_lock[drift["outcome_lock"]]
        verified_support = [
            {
                "path": row["path"],
                "sha256": row["actual_sha256"],
                "path_key": row["path_key"],
            }
            for row in source["dependency_audit"]["verified"]
            if row["path_key"] in {
                "result",
                "summary",
                "audit",
                "access",
                "design_lock",
                "evaluation_lock",
                "implementation_lock",
                "protocol_lock",
                "source_lock",
            }
        ]
        recovered = any(row["matches_frozen_expected_sha256"] for row in versions)
        current_sha = sha256_file(ROOT / path)
        current_matches_all_reachable = bool(versions) and all(row["sha256"] == current_sha for row in versions)
        findings.append(
            {
                "outcome_lock": drift["outcome_lock"],
                "outcome_lock_payload_valid": source["outcome_lock"]["payload_lock_valid"],
                "dependency_path": path,
                "dependency_role": drift["path_key"],
                "classification": _classification(path),
                "frozen_expected_sha256": expected,
                "current_sha256": current_sha,
                "current_file_matches_HEAD_and_index": status == "",
                "git_status": status,
                "reachable_or_reflog_versions": versions,
                "content_diff_diagnosis": {
                    "current_matches_every_reachable_or_reflog_version": current_matches_all_reachable,
                    "current_vs_reachable_changed_byte_count": 0 if current_matches_all_reachable else None,
                    "current_vs_reachable_changed_line_count": 0 if current_matches_all_reachable else None,
                    "frozen_expected_content_available_for_direct_diff": recovered,
                    "conclusion": (
                        "current bytes equal the only recoverable repository version; a direct content diff to the frozen expected bytes is impossible because those bytes are absent"
                        if current_matches_all_reachable and not recovered
                        else "inspect_recovered_or_multiple_versions"
                    ),
                },
                "frozen_bytes_recovered_from_git_history": recovered,
                "verified_supporting_artifacts": verified_support,
                "resolution": (
                    "eligible_for_exact_non_destructive_recovery"
                    if recovered
                    else "do_not_guess_or_overwrite_preserve_append_only_addendum"
                ),
                "claim_effect": (
                    "The frozen outcome payload remains self-valid, but exact present-worktree reconstruction of this dependency is unavailable. "
                    "Other hash-matched artifacts remain independently listed and must carry the numerical or structural claim."
                ),
            }
        )
    return {
        "schema_version": "dependency_drift_provenance_addendum.v1",
        "source_audit": str(AUDIT_PATH.relative_to(ROOT)),
        "finding_count": len(findings),
        "all_outcome_payloads_valid": all(row["outcome_lock_payload_valid"] for row in findings),
        "exact_frozen_dependency_recovery_count": sum(row["frozen_bytes_recovered_from_git_history"] for row in findings),
        "current_dirty_dependency_count": sum(not row["current_file_matches_HEAD_and_index"] for row in findings),
        "findings": findings,
        "decision": "preserve_all_eight_as_append_only_provenance_gaps_without_overwriting_current_files",
        "authorization": {
            "rewrite_historical_outcome_locks": False,
            "overwrite_current_dependencies": False,
            "claim_exact_reconstruction_for_drifted_dependency": False,
            "open_protected_or_request_language": False,
            "run_model_or_api": False,
        },
    }


def render(addendum: dict[str, Any]) -> str:
    sections = [
        "# Dependency-drift provenance addendum through V224",
        "",
        "## Decision",
        "",
        "None of the eight originally hashed dependency byte sequences is recoverable from any reachable ref or reflog. Each affected file appears in one reachable commit, already at its current hash, and every current file matches HEAD/index. Reconstructing an alleged original would therefore be guesswork. The files and historical locks remain unchanged; this append-only addendum preserves both hashes and narrows the reconstruction claim.",
        "",
        "The outcome-lock payloads themselves remain valid. A drifted verifier/auditor means its exact historical executable is unavailable; a drifted results document means the present prose is not the exact document frozen by that outcome. Hash-matched result, summary, audit, access, and design artifacts listed in the machine-readable addendum remain the appropriate support where available.",
        "",
        "## Findings",
        "",
        "| Outcome lock | Dependency | Type | Frozen hash | Current hash | Recoverable |",
        "|---|---|---|---|---|---|",
    ]
    for row in addendum["findings"]:
        sections.append(
            f"| `{row['outcome_lock']}` | `{row['dependency_path']}` | {row['classification']} | `{row['frozen_expected_sha256']}` | `{row['current_sha256']}` | {'yes' if row['frozen_bytes_recovered_from_git_history'] else 'no'} |"
        )
    sections.extend(
        [
            "",
            "## Interpretation",
            "",
            "- V78 and V79 have executable-verifier provenance gaps.",
            "- V106 has an executable-auditor provenance gap for the failed-auditor dependency.",
            "- V94 and V118-V121 have narrative-results-document provenance gaps.",
            "- These are not evidence that the outcome value changed, but they prevent claiming exact reconstruction through the drifted dependency.",
            "- Each current file is byte-identical to its sole reachable/reflog version (zero changed bytes and lines); the unavailable frozen bytes cannot be directly diffed.",
            "- Repair requires an exact content-addressed copy whose SHA-256 matches the frozen hash. Editing the current file until a plausible resemblance is reached is forbidden.",
            "",
            "The complete commit inventory and surviving hash-matched support are in `outputs/cross-track-evidence-audit-through-v224/dependency-drift-provenance-addendum.json`."
        ]
    )
    return "\n".join(sections) + "\n"


def main() -> None:
    addendum = diagnose()
    write_json(OUTPUT_PATH, addendum)
    DOCUMENT_PATH.write_text(render(addendum), encoding="utf-8")
    print(json.dumps({key: addendum[key] for key in ("finding_count", "exact_frozen_dependency_recovery_count", "current_dirty_dependency_count", "decision")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
