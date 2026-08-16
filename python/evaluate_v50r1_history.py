#!/usr/bin/env python3
"""Narrow V50 execution repair for a non-gating Decimal underflow diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal, localcontext

import evaluate_v50_history as base
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def effective_count_decimal(weights):
    with localcontext() as context:
        context.prec = 100
        entropy = -sum((weight * weight.ln() for weight in weights if weight), Decimal(0))
        return float(entropy.exp())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-lock", default="configs/v50r1-repair-lock.json")
    parser.add_argument("--corpus-seal", default="configs/v50-corpus-seal.json")
    parser.add_argument("--output-dir", default="outputs/v50r1-execution-repair/development")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.repair_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["run_repair_development_once"]:
        raise RuntimeError("V50r1 repair lock does not authorize execution")
    if file_sha256(PROJECT_ROOT / lock["source_corpus_seal"]) != lock["source_corpus_seal_sha256"]:
        raise RuntimeError("V50 source corpus seal changed after repair lock")
    if file_sha256(PROJECT_ROOT / "python/evaluate_v50r1_history.py") != lock["repair_implementation"]["python/evaluate_v50r1_history.py"]:
        raise RuntimeError("V50r1 repair evaluator changed after lock")
    base.effective_count = effective_count_decimal
    sys.argv = [
        sys.argv[0],
        "--corpus-seal", args.corpus_seal,
        "--output-dir", args.output_dir,
    ]
    base.main()


if __name__ == "__main__":
    main()
