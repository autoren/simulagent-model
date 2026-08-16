#!/usr/bin/env python3
"""Freeze the V38 implementation and authorize construction only."""
from __future__ import annotations
import argparse, hashlib, json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT

FILES=("python/v38_focus_parser.py","python/generate_v38_focus_parser.py","python/v38_evaluation.py","python/audit_v38_implementation.py","python/freeze_v38_implementation.py","python/audit_v38_corpus.py","python/seal_v38_corpus.py","python/extract_v38_features_mlx.py","python/freeze_v38_features.py","python/evaluate_v38_focus_parser.py","python/audit_and_summarize_v38.py","python/freeze_v38_outcome.py","python/test_v38_focus_parser.py","python/test_v38_evaluation.py","python/v32_language.py","python/v34_operation.py","python/v35_binding.py","python/v10_protocol.py","python/v22r2_grounding.py","python/evaluate_v30_signed_fact_language_mlx.py","python/extract_v10_features_mlx.py")
def main():
    p=argparse.ArgumentParser(); p.add_argument("--design-lock",default="configs/v38-ontology-anchored-focus-parser-lock.json"); p.add_argument("--audit",default="outputs/v38-ontology-anchored-focus-parser/implementation-audit.json"); p.add_argument("--output",default="configs/v38-implementation-lock.json"); a=p.parse_args(); design_path=(PROJECT_ROOT/a.design_lock).resolve(); audit_path=(PROJECT_ROOT/a.audit).resolve(); output=(PROJECT_ROOT/a.output).resolve()
    if output.exists(): raise RuntimeError("V38 implementation already frozen")
    design=json.loads(design_path.read_text()); audit=json.loads(audit_path.read_text())
    if not audit["passed"] or audit["design_lock_sha256"]!=file_sha256(design_path): raise RuntimeError("V38 implementation audit did not pass")
    for path in FILES:
        if not (PROJECT_ROOT/path).is_file(): raise RuntimeError(f"V38 implementation incomplete: {path}")
    v32_path=PROJECT_ROOT/"configs/v32-factorized-semantics.json"; v34_path=PROJECT_ROOT/"configs/v34-operation-interface.json"; v37_features=PROJECT_ROOT/"configs/v37-features-lock.json"; v37_outcome=PROJECT_ROOT/"configs/v37-outcome-lock.json"
    lock={"schema_version":38,"experiment":"v38_implementation_lock","design_lock":str(design_path.relative_to(PROJECT_ROOT)),"design_lock_sha256":file_sha256(design_path),"implementation_audit":str(audit_path.relative_to(PROJECT_ROOT)),"implementation_audit_sha256":file_sha256(audit_path),"config_payload":design["config_payload"],"v32_config_payload":json.loads(v32_path.read_text()),"v32_config_sha256":file_sha256(v32_path),"v34_config_payload":json.loads(v34_path.read_text()),"v34_config_sha256":file_sha256(v34_path),"v37_features_lock":str(v37_features.relative_to(PROJECT_ROOT)),"v37_features_lock_sha256":file_sha256(v37_features),"v37_outcome_lock":str(v37_outcome.relative_to(PROJECT_ROOT)),"v37_outcome_lock_sha256":file_sha256(v37_outcome),"expected_corpus_sha256":{"ontology_focus_fit":audit["dry_run"]["fit_corpus_sha256"],"ontology_focus_validation":audit["dry_run"]["validation_corpus_sha256"]},"forward_budget":audit["dry_run"],"implementation":{path:file_sha256(PROJECT_ROOT/path) for path in FILES},"authorization":{"construct_corpus":True,"model_access":False,"fit_parser":False,"score_validation":False,"v32_evaluation":False,"v28":False}}
    lock["lock_payload_sha256"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(json.dumps(lock,indent=2,sort_keys=True))
if __name__=="__main__": main()
