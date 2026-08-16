#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--supplement",default="configs/v48-reporting-supplement.json"); p.add_argument("--output",default="configs/v48-reporting-supplement-lock.json"); a=p.parse_args(); supplement=(PROJECT_ROOT/a.supplement).resolve(); output=(PROJECT_ROOT/a.output).resolve()
 if output.exists(): raise RuntimeError("V48 reporting supplement already frozen")
 c=json.loads(supplement.read_text()); design=PROJECT_ROOT/c["sourceDesignLock"]; errors=[]
 if not design.is_file(): errors.append("V48 design lock missing")
 if c["changesPrimaryGates"] or c["changesDecisionHierarchy"]: errors.append("Reporting supplement changes registered decisions")
 reporting=c["requiredReporting"]; alignment=c["requiredAlignmentAudit"]
 if reporting["statisticalUnit"]!="mechanic_episode" or reporting["bootstrapResamples"]!=10000 or reporting["bootstrapSeed"]!=4847: errors.append("Mechanic-level uncertainty reporting invalid")
 if alignment["requiredExactRate"]!=1.0 or not alignment["nonCompensatory"] or not all(value for key,value in alignment.items() if key not in ("requiredExactRate","nonCompensatory")): errors.append("Alignment audit is not exact and non-compensatory")
 if len(reporting["posteriorDiagnostics"])<6 or len(reporting["worstCases"])<5: errors.append("Required diagnostics incomplete")
 if errors: raise RuntimeError("; ".join(errors))
 lock={"schema_version":48,"experiment":"v48_reporting_supplement_lock","supplement":str(supplement.relative_to(PROJECT_ROOT)),"supplement_sha256":file_sha256(supplement),"source_design_lock":str(design.relative_to(PROJECT_ROOT)),"source_design_lock_sha256":file_sha256(design),"non_gating":True,"required_before_implementation_lock":True}; lock["lock_payload_sha256"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(json.dumps(lock,indent=2,sort_keys=True))
if __name__=="__main__": main()
