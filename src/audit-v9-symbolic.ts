import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { V8StructuredRecord } from './contracts';
import { readJson, writeJson } from './io';
import { sha256 } from './serialization';
import { allowedValuesFromV8Input, evaluateAllowedTransitions } from './v9-symbolic';

interface SymbolicLock {
  source: {
    dataset_sha256: string;
    manifest: string;
    manifest_sha256: string;
    artifacts: Record<string, string>;
  };
  config: {
    output: string;
    requiredRecords: number;
    requiredMismatches: number;
  };
}

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const lockPath = resolve(projectRoot, argumentValue('--lock') ?? 'configs/v9-symbolic-lock.json');
const lock = await readJson<SymbolicLock>(lockPath);
const manifestPath = resolve(projectRoot, lock.source.manifest);
const manifestContent = await readFile(manifestPath, 'utf8');
if (sha256(manifestContent) !== lock.source.manifest_sha256) {
  throw new Error('V9 symbolic source manifest changed after lock.');
}

const records: V8StructuredRecord[] = [];
for (const [relative, expected] of Object.entries(lock.source.artifacts)) {
  const path = resolve(projectRoot, relative);
  const content = await readFile(path, 'utf8');
  if (sha256(content) !== expected) throw new Error(`V9 symbolic source changed: ${relative}`);
  records.push(...content.trim().split('\n').filter(Boolean).map((line) => JSON.parse(line) as V8StructuredRecord));
}

let identifiabilityMismatches = 0;
let transitionCountMismatches = 0;
let assignmentCountMismatches = 0;
const cells: Record<string, {
  records: number;
  identifiability_mismatches: number;
  transition_count_mismatches: number;
  assignment_count_mismatches: number;
}> = {};
for (const record of records) {
  const result = evaluateAllowedTransitions({
    action_dependency_schema: record.agent_input.action_dependency_schema,
    determinant_values: allowedValuesFromV8Input(record.agent_input),
  });
  const identityMismatch = result.identifiable !== !record.target.ambiguous;
  const transitionMismatch = result.possible_transition_codes.length !== record.target.possible_transition_count;
  const assignmentMismatch = result.compatible_assignments !== record.oracle.compatible_assignments;
  identifiabilityMismatches += Number(identityMismatch);
  transitionCountMismatches += Number(transitionMismatch);
  assignmentCountMismatches += Number(assignmentMismatch);
  const key = `${record.mechanic}/${record.surface_variant}`;
  const cell = cells[key] ?? {
    records: 0,
    identifiability_mismatches: 0,
    transition_count_mismatches: 0,
    assignment_count_mismatches: 0,
  };
  cell.records += 1;
  cell.identifiability_mismatches += Number(identityMismatch);
  cell.transition_count_mismatches += Number(transitionMismatch);
  cell.assignment_count_mismatches += Number(assignmentMismatch);
  cells[key] = cell;
}

const totalMismatches = identifiabilityMismatches + transitionCountMismatches + assignmentCountMismatches;
if (records.length !== lock.config.requiredRecords) {
  throw new Error(`V9 symbolic audit expected ${lock.config.requiredRecords} records, found ${records.length}.`);
}
if (totalMismatches !== lock.config.requiredMismatches) {
  throw new Error(`V9 symbolic audit found ${totalMismatches} mismatches.`);
}
const report = {
  schema_version: 9,
  experiment: 'v9_symbolic_oracle_audit',
  protocol_lock: relative(lockPath),
  protocol_lock_sha256: sha256(await readFile(lockPath, 'utf8')),
  source_dataset_sha256: lock.source.dataset_sha256,
  records: records.length,
  identifiability_mismatches: identifiabilityMismatches,
  transition_count_mismatches: transitionCountMismatches,
  compatible_assignment_count_mismatches: assignmentCountMismatches,
  cells: Object.fromEntries(Object.entries(cells).sort(([left], [right]) => left.localeCompare(right))),
  passed: totalMismatches === 0,
  decision: totalMismatches === 0 ? 'authorize_v9_grounding_generation' : 'stop_before_v9_grounding',
  data_access: {
    v3_test_records_read: 0,
    prior_holdout_records_read: 0,
    v7_tone_drift_records_read: 0,
    v7_model_results_read: 0,
    untouched_v8_mechanic_records_read: 0,
    final_v9_mechanic_records_read: 0,
  },
};
const outputPath = resolve(projectRoot, lock.config.output);
await writeJson(outputPath, report);
console.log(JSON.stringify(report, null, 2));

function argumentValue(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function relative(path: string): string {
  return path.startsWith(`${projectRoot}/`) ? path.slice(projectRoot.length + 1) : path;
}
