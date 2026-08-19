from __future__ import annotations

from v220_fresh_mondo_artifact_population import audit_population


def test_event_strata_are_noncompensatory() -> None:
    config = {
        "populationGates": {
            "oracleEvaluationScope": "DEVELOPMENT_ONLY_WITH_PROTECTED_FILES_HASHED_BUT_NOT_LOADED",
            "minimumAdditionEventFamilyCount": 24,
            "minimumTextChangeFamilyCount": 4,
            "minimumLifecycleEventFamilyCount": 3,
            "minimumMappingEventFamilyCount": 12,
            "minimumAmbiguousUnspecifiedFamilyCount": 12,
            "minimumDecisionContrastFamilyCount": 12,
        },
        "accessGates": {
            "maximumV218DevelopmentRecordReadCount": 0,
            "maximumV218ProtectedRecordReadCount": 0,
            "maximumProtectedFileLoadForScoringCount": 0,
        },
        "decisionRule": {
            "ifEveryPayloadControlPopulationStratumIntegrityAndAccessGatePasses": "pass",
            "otherwise": "fail",
        },
    }
    metrics = {
        "addition_event_family_count": 1000,
        "text_change_family_count": 3,
        "lifecycle_event_family_count": 3,
        "mapping_event_family_count": 12,
        "ambiguous_unspecified_family_count": 12,
        "decision_contrast_family_count": 12,
        "oracle_evaluation_scope": "DEVELOPMENT_ONLY",
        "protected_files_loaded_for_scoring": False,
    }
    # Patch the inherited audit at its boundary: an arbitrarily large addition
    # stratum cannot make the explicit text stratum pass.
    import v220_fresh_mondo_artifact_population as module

    original = module._audit_population
    module._audit_population = lambda *_args, **_kwargs: {"checks": {"base": True}, "access_checks": {"base": True}}
    try:
        result = audit_population(
            metrics,
            {
                "v218_development_record_read_count": 0,
                "v218_protected_record_read_count": 0,
                "protected_file_load_for_scoring_count": 0,
            },
            config,
        )
    finally:
        module._audit_population = original
    assert not result["passed"]
    assert not result["checks"]["event_strata_are_direct_and_noncompensatory"]
