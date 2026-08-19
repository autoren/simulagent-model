from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cross-track-evidence-audit-through-v224.json"
OUTPUT_DIR = ROOT / "outputs/cross-track-evidence-audit-through-v224"
SYNTHESIS_PATH = ROOT / "docs/cross-track-evidence-synthesis-through-v224.md"
STOPPING_PATH = ROOT / "docs/research-stopping-rule-after-v224.md"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def valid_lock(lock: dict[str, Any]) -> bool:
    expected = lock.get("lock_payload_sha256")
    if not isinstance(expected, str):
        return False
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    return payload_hash(payload) == expected


def version_number(path_or_name: str) -> int:
    match = re.search(r"(?:^|/)v(\d+)", path_or_name)
    if not match:
        raise ValueError(f"No version number in {path_or_name!r}")
    return int(match.group(1))


def version_token(path_or_name: str) -> str:
    name = Path(path_or_name).name
    match = re.match(r"(v\d+(?:r\d+|[a-z])?)", name)
    if not match:
        raise ValueError(f"No version token in {path_or_name!r}")
    return match.group(1)


def is_repair(path_or_name: str) -> bool:
    name = Path(path_or_name).name.lower()
    return bool(re.match(r"v\d+r\d+", name)) or "repair" in name


def _walk_dict_scalars(value: Any, path: str = "", depth: int = 0) -> Iterable[tuple[str, str, Any]]:
    if not isinstance(value, dict) or depth > 8:
        return
    for key, item in value.items():
        item_path = f"{path}.{key}" if path else key
        if isinstance(item, (str, int, float, bool)) or item is None:
            yield item_path, key, item
        elif isinstance(item, dict):
            yield from _walk_dict_scalars(item, item_path, depth + 1)


def outcome_signals(lock: dict[str, Any]) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    qualification_flags: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path, key, value in _walk_dict_scalars(lock):
        lower = key.lower()
        bucket: list[dict[str, Any]] | None = None
        if lower == "decision" or lower == "scientific_decision" or lower.endswith("_decision"):
            bucket = decisions
        elif lower == "branch" or lower.endswith("_branch"):
            bucket = branches
        elif (
            lower in {"passed", "qualified", "development_qualified", "policy_qualified", "condition_qualified"}
            or ("scientific" in lower and "passed" in lower)
            or lower.endswith("_qualified")
            or lower.startswith("passed_all_")
        ) and isinstance(value, bool):
            bucket = qualification_flags
        if bucket is not None:
            marker = (path, str(value))
            if marker not in seen:
                seen.add(marker)
                bucket.append({"path": path, "value": value})
    return {
        "decisions": decisions[:24],
        "branches": branches[:12],
        "qualification_flags": qualification_flags[:32],
    }


def classify_status(signals: dict[str, Any], repair: bool) -> str:
    texts = " ".join(str(row["value"]).lower() for row in signals["decisions"] + signals["branches"])
    flags = signals["qualification_flags"]
    scientific_false = any(
        row["value"] is False
        and any(
            token in row["path"].lower()
            for token in ("scientific", "qualified", "passed_all", "development_qualified", "policy_qualified")
        )
        for row in flags
    )
    negative_text = bool(
        re.search(
            r"(^|_)(negative|fail(?:s|ed|ure)?|reject|nonqualifying|nonqualif|infeasible|does_not|stop|close)(_|$)",
            texts,
        )
    )
    positive_text = bool(re.search(r"(^|_)(positive|confirm|accept|qualif|pass|feasible|sufficient)(_|$)", texts))
    if negative_text or scientific_false:
        scientific = "negative_or_boundary"
    elif positive_text or any(row["value"] is True for row in flags):
        scientific = "positive_or_qualified"
    else:
        scientific = "unclassified_from_generic_schema"
    return f"repair__{scientific}" if repair else scientific


def _path_hash_pairs(value: Any, object_path: str = "") -> Iterable[dict[str, Any]]:
    if not isinstance(value, dict):
        return
    for key, expected in value.items():
        if key.endswith("_sha256") and isinstance(expected, str):
            base = key[: -len("_sha256")]
            candidate = value.get(base)
            if isinstance(candidate, str):
                yield {
                    "object_path": object_path,
                    "path_key": base,
                    "hash_key": key,
                    "path": candidate,
                    "expected_sha256": expected,
                }
    for key, item in value.items():
        if isinstance(item, dict):
            child = f"{object_path}.{key}" if object_path else key
            yield from _path_hash_pairs(item, child)


def _dependency_sensitivity(pair: dict[str, Any], config: dict[str, Any]) -> str | None:
    raw_path = pair["path"]
    if raw_path.startswith(("http://", "https://", "git://")):
        return "remote_identifier_not_local_dependency"
    path = Path(raw_path)
    suffixes = set(config["dependency_hash_policy"]["skip_extensions"])
    if path.suffix.lower() in suffixes:
        return "protected_or_record_level_extension"
    lower_parts = {part.lower() for part in path.parts}
    forbidden_parts = set(config["dependency_hash_policy"]["skip_path_components"])
    if lower_parts & forbidden_parts:
        return "protected_raw_or_holdout_path"
    key = pair["path_key"].lower()
    if any(token in key for token in ("protected_record", "raw_record", "request_body", "request_language")):
        return "protected_or_language_key"
    return None


def audit_dependencies(lock: dict[str, Any], root: Path, config: dict[str, Any]) -> dict[str, Any]:
    verified: list[dict[str, Any]] = []
    drifted: list[dict[str, Any]] = []
    living_document_drift: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for pair in _path_hash_pairs(lock):
        marker = (pair["path"], pair["expected_sha256"], pair["path_key"])
        if marker in seen:
            continue
        seen.add(marker)
        reason = _dependency_sensitivity(pair, config)
        if reason:
            skipped.append({**pair, "reason": reason})
            continue
        candidate = Path(pair["path"])
        if candidate.is_absolute():
            try:
                candidate.relative_to(root)
            except ValueError:
                skipped.append({**pair, "reason": "absolute_path_outside_repository"})
                continue
        else:
            candidate = root / candidate
        if not candidate.exists() or not candidate.is_file():
            missing.append({**pair, "reason": "referenced_local_file_missing"})
            continue
        actual = sha256_file(candidate)
        row = {**pair, "actual_sha256": actual}
        if actual == pair["expected_sha256"]:
            verified.append(row)
        elif pair["path"] in set(config["dependency_hash_policy"]["living_documents_expected_to_change"]):
            living_document_drift.append(
                {**row, "reason": "append-only research direction document intentionally changed after this historical outcome"}
            )
        else:
            drifted.append(row)
    return {
        "verified": verified,
        "drifted": drifted,
        "living_document_drift": living_document_drift,
        "skipped": skipped,
        "missing": missing,
    }


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _lock_record(root: Path, path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return {
        "path": _relative(root, path),
        "file_sha256": sha256_file(path),
        "payload_lock_valid": valid_lock(payload),
        "declared_payload_sha256": payload.get("lock_payload_sha256"),
    }


def associated_design_locks(
    root: Path, outcome_path: Path, outcome: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    referenced_paths: set[str] = set()
    for pair in _path_hash_pairs(outcome):
        path = pair["path"]
        if "lock" in pair["path_key"].lower() and path.startswith("configs/") and path.endswith(".json"):
            referenced_paths.add(path)
    referenced: list[dict[str, Any]] = []
    for relative in sorted(referenced_paths):
        candidate = root / relative
        if candidate.exists() and candidate.is_file():
            referenced.append(_lock_record(root, candidate))
        else:
            referenced.append({"path": relative, "missing": True})

    token = version_token(outcome_path.name)
    same_version: list[dict[str, Any]] = []
    referenced_set = {row["path"] for row in referenced}
    for candidate in sorted((root / "configs").glob(f"{token}*-lock.json")):
        if candidate.name.endswith("-outcome-lock.json"):
            continue
        relative = _relative(root, candidate)
        if relative not in referenced_set:
            same_version.append(_lock_record(root, candidate))
    return referenced, same_version


def _signal_map_from_locks(root: Path, rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    patterns = {
        "claim": ("claim", "boundary", "non_claim", "interpretation", "validity", "scope"),
        "evidence": ("source", "dataset", "corpus", "provenance", "evidence"),
    }[kind]
    signals: list[dict[str, Any]] = []
    for row in rows:
        if row.get("missing"):
            continue
        payload = read_json(root / row["path"])
        for path, key, value in _walk_dict_scalars(payload):
            if any(pattern in key.lower() for pattern in patterns):
                if isinstance(value, str) and len(value) > 500:
                    value = value[:497] + "..."
                signals.append({"lock": row["path"], "path": path, "value": value})
                if len(signals) >= 40:
                    return signals
    return signals


def _model_activity(lock: dict[str, Any], outcome_path: Path) -> dict[str, Any]:
    counters: list[dict[str, Any]] = []
    for path, key, value in _walk_dict_scalars(lock):
        lower = key.lower()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and any(
            token in lower
            for token in (
                "model_call_count",
                "model_generation_count",
                "model_load_count",
                "api_call_count",
                "training_run_count",
                "local_generation_count",
            )
        ):
            counters.append({"path": path, "value": value})
    return {
        "declared_counters": counters[:24],
        "declared_nonzero_activity": any(row["value"] > 0 for row in counters),
        "experiment_label_mentions_model": any(
            token in outcome_path.name.lower() for token in ("model", "llm", "candidate-generation", "rank-only")
        ),
    }


def _authorization(lock: dict[str, Any]) -> Any:
    value = lock.get("authorization")
    return value if isinstance(value, (dict, list, str, bool)) else None


def audit_repository(root: Path = ROOT, config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = read_json(config_path)
    first = config["scope"]["first_version"]
    last = config["scope"]["last_version"]
    families = config["families"]
    outcomes = sorted(
        (
            path
            for path in (root / "configs").glob("v*-outcome-lock.json")
            if first <= version_number(path.name) <= last
        ),
        key=lambda path: (version_number(path.name), path.name),
    )

    family_by_version: dict[int, dict[str, Any]] = {}
    for family in families:
        for number in range(family["version_min"], family["version_max"] + 1):
            if number in family_by_version:
                raise AssertionError(f"Overlapping family assignment for V{number}")
            family_by_version[number] = family

    experiment_rows: list[dict[str, Any]] = []
    for path in outcomes:
        lock = read_json(path)
        number = version_number(path.name)
        family = family_by_version.get(number)
        if family is None:
            raise AssertionError(f"No family assignment for {path.name}")
        repair = is_repair(path.name)
        signals = outcome_signals(lock)
        dependencies = audit_dependencies(lock, root, config)
        referenced, same_version = associated_design_locks(root, path, lock)
        all_design = referenced + same_version
        docs = [
            pair
            for pair in _path_hash_pairs(lock)
            if isinstance(pair["path"], str) and pair["path"].startswith("docs/")
        ]
        experiment_rows.append(
            {
                "version_number": number,
                "version_token": version_token(path.name),
                "family_id": family["family_id"],
                "outcome_lock": _lock_record(root, path),
                "experiment": lock.get("experiment"),
                "schema_version": lock.get("schema_version"),
                "repair": repair,
                "status_class": classify_status(signals, repair),
                "outcome_signals": signals,
                "authorization": _authorization(lock),
                "referenced_design_locks": referenced,
                "same_version_unreferenced_design_locks": same_version,
                "claim_boundary_signals": _signal_map_from_locks(root, all_design, "claim"),
                "evidence_source_signals": _signal_map_from_locks(root, all_design, "evidence"),
                "referenced_results_documents": docs,
                "model_involvement": _model_activity(lock, path),
                "dependency_audit": dependencies,
            }
        )

    covered_numbers = {row["version_number"] for row in experiment_rows}
    missing_versions = [number for number in range(first, last + 1) if number not in covered_numbers]
    expected_missing = config["scope"]["known_versions_without_frozen_outcome"]
    if missing_versions != expected_missing:
        raise AssertionError(f"Frozen missing-version set changed: {missing_versions} != {expected_missing}")

    family_rows: list[dict[str, Any]] = []
    for family in families:
        members = [row for row in experiment_rows if row["family_id"] == family["family_id"]]
        family_rows.append(
            {
                **family,
                "outcome_lock_count": len(members),
                "outcome_locks": [row["outcome_lock"]["path"] for row in members],
                "design_locks": sorted(
                    {
                        design["path"]
                        for row in members
                        for design in row["referenced_design_locks"] + row["same_version_unreferenced_design_locks"]
                    }
                ),
                "repair_outcome_locks": [row["outcome_lock"]["path"] for row in members if row["repair"]],
                "status_class_counts": dict(sorted(Counter(row["status_class"] for row in members).items())),
                "dependency_drift_count": sum(len(row["dependency_audit"]["drifted"]) for row in members),
                "missing_dependency_count": sum(len(row["dependency_audit"]["missing"]) for row in members),
            }
        )

    by_path = {row["outcome_lock"]["path"]: row for row in experiment_rows}
    critical_rows: list[dict[str, Any]] = []
    for index, item in enumerate(config["critical_chain"], start=1):
        source = by_path.get(item["outcome_lock"])
        if source is None:
            raise AssertionError(f"Critical outcome missing: {item['outcome_lock']}")
        critical_rows.append(
            {
                "sequence": index,
                **item,
                "family_id": source["family_id"],
                "outcome_file_sha256": source["outcome_lock"]["file_sha256"],
                "payload_lock_valid": source["outcome_lock"]["payload_lock_valid"],
                "status_class": source["status_class"],
                "design_locks": [row["path"] for row in source["referenced_design_locks"]],
                "dependency_drift": source["dependency_audit"]["drifted"],
                "dependency_missing": source["dependency_audit"]["missing"],
                "link_to_previous": (
                    "conceptual_composition_only; no direct cross-experiment lock reference is claimed"
                    if index > 1
                    else "chain_root"
                ),
            }
        )

    all_drift = [
        {"outcome_lock": row["outcome_lock"]["path"], **drift}
        for row in experiment_rows
        for drift in row["dependency_audit"]["drifted"]
    ]
    all_missing = [
        {"outcome_lock": row["outcome_lock"]["path"], **missing}
        for row in experiment_rows
        for missing in row["dependency_audit"]["missing"]
    ]
    all_skipped = [
        {"outcome_lock": row["outcome_lock"]["path"], **skipped}
        for row in experiment_rows
        for skipped in row["dependency_audit"]["skipped"]
    ]
    all_living_document_drift = [
        {"outcome_lock": row["outcome_lock"]["path"], **drift}
        for row in experiment_rows
        for drift in row["dependency_audit"]["living_document_drift"]
    ]
    verified_count = sum(len(row["dependency_audit"]["verified"]) for row in experiment_rows)

    roadmap_paths = sorted((root / "docs").glob("research-roadmap-after-v*.md"))
    historical_roadmaps = [
        {
            "path": _relative(root, path),
            "status": (
                "canonical_current_roadmap" if path.name == "research-roadmap-after-v224.md" else "historical_snapshot_not_current_authorization"
            ),
            "canonical_current_roadmap": "docs/research-roadmap-after-v224.md",
        }
        for path in roadmap_paths
    ]
    claim_matrix = {
        "known_cross_track_risks": config["known_cross_track_risks"],
        "historical_roadmaps": historical_roadmaps,
        "stale_or_superseded_top_level_statements": [
            {
                "path": "docs/research-direction.md",
                "finding": "Its leading current-transition section stops at V206 and later names V202 as canonical; V224 and this audit supersede that navigation.",
                "repair": "Add a leading audit-era canonical status block without deleting the historical transition record."
            },
            {
                "path": "docs/open-world-language-research-direction.md",
                "finding": "Its V224 section is current, while the many older 'current update' sections are historical snapshots.",
                "repair": "Add a leading audit synthesis/stopping-rule pointer."
            },
            {
                "path": "docs/research-roadmap-after-v223.md",
                "finding": "Its conditional V225 authorization is explicitly superseded by V224's failed record-level gate.",
                "repair": "Treat as historical only; do not edit the preregistered historical document."
            }
        ],
        "unsupported_generalizations": [
            "LLM rank, confidence, agreement, or reasoning length is not a semantic likelihood or posterior.",
            "Finite-menu proposal recall is not unrestricted open-world recognition or calibrated abstention.",
            "Controlled/synthetic mechanism success is not prospective external-language validation.",
            "Retrospective catalog reconstruction is not evidence of a new speaker's intended concept.",
            "Workflow-level human curation is not record-level four-way semantic gold.",
            "A verifier or transport repair is not an independent scientific replication."
        ]
    }

    gate = {
        "nonduplicative_unresolved_premise": {
            "passed": True,
            "finding": "A prospectively grounded record-level semantic observation channel remains genuinely unresolved."
        },
        "independent_evidence_source_already_obtainable": {
            "passed": False,
            "finding": "V224 found zero qualifying records in every frozen substantive stratum; no other audited source passes the required workflow dimensions."
        },
        "prospectively_frozen_record_level_feasibility_gate": {
            "passed": False,
            "finding": "The only eligible workflow failed the V224 preliminary record-level gate."
        },
        "observable_gold_without_model_or_simulated_human_judgment": {
            "passed": False,
            "finding": "No currently audited source exposes the full four-way semantics with independent record-level adjudication."
        },
        "meaningful_deterministic_residual": {
            "passed": None,
            "finding": "Cannot be assessed prospectively before an admissible independently grounded population exists; V214 and V221 both close their existing tasks deterministically."
        }
    }
    authorized = 1 if all(item["passed"] is True for item in gate.values()) else 0
    stopping = {
        "gate_results": gate,
        "authorized_next_experiment_count": authorized,
        "decision": (
            "authorize_one_prospectively_frozen_experiment" if authorized else "authorize_zero_and_freeze_experimental_escalation"
        ),
        "unresolved_premise": "Prospectively grounded record-level semantic observations that distinguish known, novel-valid, insufficient, and unsupported states and can support calibrated decision likelihoods.",
        "reopen_only_if": [
            "a new source already contains immutable requester language, explicit per-record independent adjudication, catalog/version context, and adequate frozen strata",
            "qualified speakers or domain experts become available under a prospective role-separated protocol",
            "or the question is narrowed prospectively to an outcome an available source actually records"
        ]
    }

    reproducibility = {
        "outcome_lock_count": len(experiment_rows),
        "payload_valid_count": sum(row["outcome_lock"]["payload_lock_valid"] for row in experiment_rows),
        "payload_invalid": [row["outcome_lock"]["path"] for row in experiment_rows if not row["outcome_lock"]["payload_lock_valid"]],
        "verified_non_sensitive_dependency_pair_count": verified_count,
        "drifted_non_sensitive_dependency_pair_count": len(all_drift),
        "drifted_non_sensitive_dependencies": all_drift,
        "expected_living_document_drift_count": len(all_living_document_drift),
        "expected_living_document_drift": all_living_document_drift,
        "missing_non_sensitive_dependency_pair_count": len(all_missing),
        "missing_non_sensitive_dependencies": all_missing,
        "skipped_sensitive_or_nonlocal_dependency_pair_count": len(all_skipped),
        "skipped_sensitive_or_nonlocal_dependencies": all_skipped,
        "versions_without_frozen_outcome": missing_versions,
        "protected_body_read_count": 0,
        "request_language_read_count": 0,
        "model_or_api_run_count": 0,
    }
    return {
        "audit_id": config["audit_id"],
        "frozen_at_utc": config["frozen_at_utc"],
        "experiment_ledger": experiment_rows,
        "family_ledger": family_rows,
        "reproducibility_audit": reproducibility,
        "critical_chain": critical_rows,
        "claim_and_risk_matrix": claim_matrix,
        "stopping_decision": stopping,
    }


def render_synthesis(audit: dict[str, Any]) -> str:
    rep = audit["reproducibility_audit"]
    families = audit["family_ledger"]
    drift_lines = []
    for row in rep["drifted_non_sensitive_dependencies"]:
        drift_lines.append(
            f"- `{row['outcome_lock']}` -> `{row['path']}`: expected `{row['expected_sha256']}`, found `{row['actual_sha256']}`."
        )
    if not drift_lines:
        drift_lines = ["- None."]
    family_rows = [
        f"| {row['family_id']} | V{row['version_min']}-V{row['version_max']} | {row['status']} | {row['continuation']} |"
        for row in families
    ]
    return "\n".join(
        [
            "# Cross-track evidence synthesis through V224/V224r2",
            "",
            "## Bottom line",
            "",
            "The project has a coherent uncertainty-aware decision architecture at the mechanism level, but it still lacks one empirical input: an independently grounded, prospective record-level semantic observation channel. Exact planning, typed clarification, reversible sandboxing, certificates, robust evidence gathering, explicit outside-semantics hypotheses, and safe deferral compose. Local-model proposal studies, retrospective Mondo reconstruction, and workflow/source censuses remain useful but do not supply that missing likelihood or gold standard.",
            "",
            "No next experiment is authorized. Under the frozen gate, the unresolved premise is nonduplicative, but no admissible independent source is already obtainable: V224 populated none of the four required semantic strata. Further benchmark/model escalation would repeat the premise rather than test it.",
            "",
            "## Audit coverage",
            "",
            f"- Frozen outcome locks audited: **{rep['outcome_lock_count']}**.",
            f"- Canonical payload hashes valid: **{rep['payload_valid_count']} / {rep['outcome_lock_count']}**.",
            f"- Non-sensitive referenced dependency pairs verified: **{rep['verified_non_sensitive_dependency_pair_count']}**.",
            f"- Non-sensitive dependency drifts: **{rep['drifted_non_sensitive_dependency_pair_count']}**.",
            f"- Expected append-only direction-document drifts (reported separately): **{rep['expected_living_document_drift_count']}**.",
            f"- Missing non-sensitive dependencies: **{rep['missing_non_sensitive_dependency_pair_count']}**.",
            f"- Sensitive/nonlocal pairs deliberately not opened: **{rep['skipped_sensitive_or_nonlocal_dependency_pair_count']}**.",
            f"- Versions without a frozen outcome lock: **{', '.join('V'+str(v) for v in rep['versions_without_frozen_outcome'])}**.",
            "- Protected bodies, request language, and model/API runs during the audit: **0 / 0 / 0**.",
            "",
            "A missing outcome version is a ledger gap, not evidence of failure. A valid outcome lock with a changed dependency remains valid as a historical payload, but its present-worktree reconstruction is no longer exact.",
            "",
            "## Reproducibility drift preserved",
            "",
            *drift_lines,
            "",
            "These files were not rewritten. Silently updating expected hashes would erase provenance; restoration requires recovering the originally hashed dependency or freezing an explicit addendum.",
            "",
            "## Family-level disposition",
            "",
            "| Family | Versions | Evidence status | Disposition |",
            "|---|---:|---|---|",
            *family_rows,
            "",
            "Full claims, exact member outcome/design locks, repairs, protected boundaries, model involvement, and unresolved premises are in `outputs/cross-track-evidence-audit-through-v224/family-ledger.json`.",
            "",
            "## What composes into the architecture",
            "",
            "1. **Represent hypotheses and equivalence safely.** V212-V214 provide exact expressibility/evidence states, equivalence collapse, contradiction handling, and a rule that model candidates remain provisional.",
            "2. **Optionally propose, never authorize.** V195/V198 support a narrow finite-menu local proposal role; V201 and V202 require presentation-aware normalization and a trusted controller retaining the full hypothesis set.",
            "3. **Maintain beliefs and an outside regime.** V50r1/V51r1 and V205/V209 supply history-dependent inference plus an explicit outside-semantics state. Model confidence is not substituted for a likelihood.",
            "4. **Choose evidence for decision value.** V63-V75, V116-V122, and V183-V190 define when sensing/clarification can change later control, including binary/multiway channel limits and safe OTHER/defer outcomes.",
            "5. **Sandbox provisional consequences.** V168/V171 retain reversible transaction semantics while V175/V177 and V180/V182 add certificate-aware and one-corruption-robust evidence policies.",
            "6. **Act only after settlement; otherwise defer.** V205's terminally proper structure prevents horizon escape and makes calibration, inspection, state-specific repair, or safe deferral optimal on different histories.",
            "",
            "This is conceptual composition across separately frozen studies. The ledger does not claim that every later lock directly references every earlier one, nor that the complete stack has been externally validated end to end.",
            "",
            "## What does not compose as semantic evidence",
            "",
            "- LLM rank, confidence, agreement, or longer reasoning traces cannot be used as posterior mass.",
            "- Finite-menu proposal recall cannot be reported as unrestricted open-world recognition.",
            "- A synthetic person/model answer cannot replace independent speaker or expert adjudication.",
            "- Retrospective versioned ontology reconstruction cannot identify a new speaker's intended concept.",
            "- Mondo's workflow-level human curation cannot replace the absent record-level four-way population.",
            "- Transport, parser, metric, or verifier repairs cannot be counted as independent replications.",
            "",
            "## Duplicate and post-hoc risk",
            "",
            "The main multiplicity risk is the sequence of related local-model recovery/interface variants across V80-V92, V107-V115, and V132-V164. Those runs diagnose different interfaces but repeatedly encounter the same missing premise: independently observable open-world semantic boundaries. They should be synthesized as bounded diagnostics, not counted as many independent tests of model competence. The scientific replications that do count as fresh mechanism confirmation are the role-separated/procedural confirmations such as V70, V75, V170/V171, V177, and V182.",
            "",
            "## External evidence boundary",
            "",
            "V221 is a legitimate positive external retrospective result: immutable Mondo artifacts plus deterministic exact-family expansion solve catalog-version reconstruction. V223 is a legitimate positive workflow census. V224 then narrows both: the same source does not yield a usable prospective record-level known/new/insufficient/unsupported population under the frozen metadata gate. The combined result is not contradictory; it separates catalog reconstruction from speaker semantics.",
            "",
            "## Canonical next state",
            "",
            "Experimental escalation is frozen. Continue only documentation, reproducibility restoration, and architecture consolidation until an external-state change satisfies the reopening rule in `docs/research-stopping-rule-after-v224.md`."
        ]
    ) + "\n"


def render_stopping_rule(audit: dict[str, Any]) -> str:
    stop = audit["stopping_decision"]
    gate_rows = []
    for name, row in stop["gate_results"].items():
        state = "PASS" if row["passed"] is True else "FAIL" if row["passed"] is False else "NOT TESTABLE"
        gate_rows.append(f"| `{name}` | {state} | {row['finding']} |")
    reopen = [f"- {item}" for item in stop["reopen_only_if"]]
    return "\n".join(
        [
            "# Research stopping rule after V224 and the cross-track audit",
            "",
            "## Decision",
            "",
            "**Authorize zero new experiments and freeze experimental/model escalation.**",
            "",
            "The maximum was one, but every frozen condition had to pass. The project has a real unresolved premise; it does not currently have admissible independent evidence with which to test it.",
            "",
            "## Gate assessment",
            "",
            "| Mandatory condition | Result | Evidence |",
            "|---|---|---|",
            *gate_rows,
            "",
            "The untestable residual gate is not treated as a pass. V214 and V221 already show that the current self-contained and retrospective tasks close deterministically; inventing a new residual would be post-hoc.",
            "",
            "## The unresolved premise",
            "",
            stop["unresolved_premise"],
            "",
            "This premise is upstream of model comparison. A larger local model or API cannot create independently valid labels, reveal hidden catalog-version distinctions absent from the utterance, or turn similarity into membership. Therefore no Qwen scaling, reasoning-budget, ensemble, or API study is currently justified.",
            "",
            "## Reopening conditions",
            "",
            "Reopen B2c only if at least one external condition changes:",
            "",
            *reopen,
            "",
            "After that change—and before opening language—freeze source revision, sampling frame, requester/adjudicator independence, exact semantic strata, role separation, contamination controls, minimum counts, and provenance. Then reuse V212-V214 to establish identifiability and exhaust deterministic controls. A model is eligible only if a meaningful residual survives, and its endpoint remains incremental oracle-class recall at a fixed candidate budget.",
            "",
            "## Work that remains allowed",
            "",
            "- restore or annotate the eight dependency drifts without rewriting historical claims;",
            "- consolidate a reference implementation of the composable model-free architecture;",
            "- improve documentation/navigation and mark old roadmaps as historical snapshots;",
            "- monitor for a genuinely admissible external source or availability of qualified speakers/experts; and",
            "- perform no protected-data, request-language, model/API, registration, action, or execution work under this stopping rule.",
            "",
            "Monitoring an external state change is not a new experiment. If nothing changes, stopping is the scientifically correct result."
        ]
    ) + "\n"


def build_manifest(root: Path, artifact_paths: list[Path]) -> dict[str, Any]:
    return {
        "schema_version": "cross_track_evidence_audit_manifest.v1",
        "artifacts": [
            {"path": _relative(root, path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(artifact_paths)
        ],
    }


def run_and_write(root: Path = ROOT, config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    audit = audit_repository(root, config_path)
    output_dir = root / "outputs/cross-track-evidence-audit-through-v224"
    paths = {
        "experiment_ledger": output_dir / "experiment-ledger.json",
        "family_ledger": output_dir / "family-ledger.json",
        "reproducibility_audit": output_dir / "reproducibility-audit.json",
        "critical_chain": output_dir / "critical-chain.json",
        "claim_and_risk_matrix": output_dir / "claim-and-risk-matrix.json",
        "stopping_decision": output_dir / "stopping-decision.json",
    }
    for key, path in paths.items():
        write_json(path, audit[key])
    synthesis_path = root / "docs/cross-track-evidence-synthesis-through-v224.md"
    stopping_path = root / "docs/research-stopping-rule-after-v224.md"
    synthesis_path.write_text(render_synthesis(audit), encoding="utf-8")
    stopping_path.write_text(render_stopping_rule(audit), encoding="utf-8")
    source_paths = [
        config_path,
        root / "docs/cross-track-evidence-audit-plan.md",
        root / "python/cross_track_evidence_audit.py",
        root / "python/run_cross_track_evidence_audit.py",
        root / "python/test_cross_track_evidence_audit.py",
        root / "python/verify_and_freeze_cross_track_evidence_audit.py",
    ]
    manifest = build_manifest(root, list(paths.values()) + [synthesis_path, stopping_path] + source_paths)
    write_json(output_dir / "manifest.json", manifest)
    return audit
