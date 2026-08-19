#!/usr/bin/env python3
from __future__ import annotations
import json
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v201r2_repair_decision_label_verification import evaluate_repair
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    lp=PROJECT_ROOT/"configs/v201r2-repair-decision-label-verification-lock.json"; l=json.loads(lp.read_text())
    if not valid_lock(l): raise RuntimeError("invalid V201r2 lock")
    for k in [k for k in l if not k.endswith("_sha256") and f"{k}_sha256" in l]:
        if file_sha256(PROJECT_ROOT/l[k])!=l[f"{k}_sha256"]: raise RuntimeError(k)
    out=PROJECT_ROOT/"outputs/v201r2-repair-decision-label-verification/repair"
    if out.exists(): raise RuntimeError("V201r2 already run")
    c=l["config_payload"]; r=evaluate_repair(json.loads((PROJECT_ROOT/l["source_failed_outcome_audit"]).read_text()),json.loads((PROJECT_ROOT/l["source_repair_result"]).read_text()),json.loads((PROJECT_ROOT/l["source_V201_result"]).read_text()),c); decision=c["decisionRule"]["ifExactDecisionOverwriteAndEverySubstantiveRepairCheckPasses" if r["passed"] else "otherwise"]; result={"schema_version":"201r2-repair-result","experiment":c["experiment"],"passed":r["passed"],"decision":decision,"claim_boundary":c["claimBoundary"],"repair":r,"source_artifact_mutation_count":0,"model_policy_or_scoring_rerun_count":0,"raw_model_response_read_count":0,"API_call_count":0,"actual_execution_count":0}; out.mkdir(parents=True); (out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))
    if not r["passed"]: raise SystemExit(1)


if __name__=="__main__": main()
