from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit_source_selection(config: dict[str, Any]) -> dict[str, Any]:
    massive = config["massive"]
    presto = config["presto"]
    decision = config["decision"]
    gates = config["selectionGates"]
    access = config["access"]
    authorization = config["authorization"]
    checks = {
        "core_source_is_typed_and_large_enough": bool(
            decision["coreSource"] == "MASSIVE_1.1_en-US"
            and massive["selectedLocale"] == "en-US"
            and massive["publishedDomainCount"] >= gates["minimumCoreDomainCount"]
            and massive["publishedIntentCount"] >= gates["minimumCoreIntentCount"]
            and massive["publishedSlotTypeCount"] >= gates["minimumCoreSlotTypeCount"]
            and set(massive["typedOntology"]) == {"scenario", "intent", "slot_type"}
        ),
        "core_source_has_required_labels_and_human_English_provenance": bool(
            {"scenario", "intent", "utt", "annot_utt"} <= set(massive["requiredFields"])
            and "human-created English" in massive["provenance"]
            and gates["requireHumanAuthoredCoreEnglish"]
        ),
        "insufficiency_source_is_paired_human_context_not_source_class": bool(
            decision["insufficientEvidenceSource"] == "PRESTO_v1_en-US_human_context_pairs"
            and not decision["mixSourceIdentityAsClassFeature"]
            and presto["selectedLocale"] == "en-US"
            and presto["pairRules"]["requireHumanContext"]
            and presto["pairRules"]["requireNonemptyContext"]
            and presto["pairRules"]["requireTargetArgumentAbsentFromInputAndPresentInHumanContext"]
            and presto["pairRules"]["fullAndAblatedShareUtteranceTargetAndIdentifier"]
            and not presto["pairRules"]["syntheticContextAllowed"]
        ),
        "both_artifacts_are_exactly_metadata_pinned": bool(
            massive["archive"]["byteSize"] == 40251390
            and massive["archive"]["etag"] == "51e0da2a3ff7a016f109e1d1b4306e93-3"
            and presto["archive"]["byteSize"] == 415990813
            and presto["archive"]["md5Hex"] == "5fb5bd7e437a07fbae4991b5b4a573f4"
            and presto["archive"]["gcsGeneration"] == "1678604196509246"
        ),
        "licenses_are_compatible": bool(
            massive["license"] == "CC-BY-4.0"
            and presto["license"] == "CC-BY-4.0"
            and gates["requireCCBYCompatibleLicenseForBoth"]
        ),
        "synthetic_language_is_not_external_evidence": not decision["syntheticLanguageCountsAsExternalEvaluation"],
        "rejected_sources_have_explicit_reasons": set(config["rejected"]) == {"SGD", "CLINC150", "PRESTO_as_core"},
        "zero_payload_language_model_API_training_service_or_side_effect_access": bool(
            access["payloadDownloadCount"] == gates["maximumPayloadDownloadCount"] == 0
            and access["languageRecordInspectionCount"] == gates["maximumLanguageRecordInspectionCount"] == 0
            and access["modelLoadCount"] == gates["maximumModelLoadCount"] == 0
            and access["modelGenerationCount"] == gates["maximumModelGenerationCount"] == 0
            and access["LLMAPICallCount"] == gates["maximumLLMAPICallCount"] == 0
            and access["adapterTrainingRunCount"] == gates["maximumAdapterTrainingRunCount"] == 0
            and access["realServiceCallCount"] == gates["maximumRealServiceCallCount"] == 0
            and access["externalSideEffectCount"] == gates["maximumExternalSideEffectCount"] == 0
        ),
        "authorization_stays_pre_download_and_non_authoritative": bool(
            authorization["preregisterMassiveArchiveInventory"]
            and authorization["preregisterPrestoArchiveInventory"]
            and not authorization["downloadEitherPayloadBeforeItsOwnLock"]
            and not authorization["extractOrInspectLanguage"]
            and not authorization["selectPopulation"]
            and not authorization["loadLocalOrAPIModel"]
            and not authorization["trainAdapterOrLearnLikelihood"]
            and not authorization["grantModelAuthority"]
            and not authorization["performRealServiceCallOrExternalSideEffect"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "selection_payload_sha256": canonical_sha256(config),
    }


__all__ = ["audit_source_selection", "canonical_sha256"]
