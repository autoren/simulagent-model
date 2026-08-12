import { execFileSync } from 'node:child_process';
import { readFile, rm } from 'node:fs/promises';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { V8StructuredRecord } from './contracts';
import type { V9GroundingRecord, V9OperatorFamily, V9TemplateFamily } from './v9-contracts';
import { buildV9GroundingRecords, v9OperatorByMechanic, v9TemplateFamilies } from './v9-grounding';
import { validateV9Grounding } from './v9-validation';
import { readJson, writeJson, writeJsonl } from './io';
import { canonicalJson, sha256 } from './serialization';

interface DatasetV9Config {
  outputDir: string;
  sourceManifest: string;
  sourceRecords: string[];
  symbolicLock: string;
  symbolicAudit: string;
  sourceReplicas: number[];
  contextCalibrationModulo: number;
  templateFamilies: V9TemplateFamily[];
  operatorFamilies: Record<V9OperatorFamily, string[]>;
  protocol: Record<string, unknown>;
}

interface GroundingLock {
  config_sha256: string;
  symbolic_audit_sha256: string;
  implementation: Record<string, string>;
}

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const simulatorRoot = resolve(projectRoot, '../simulagent');
const configPath = resolve(projectRoot, argumentValue('--config') ?? 'configs/dataset.v9.json');
const configContent = await readFile(configPath, 'utf8');
const config = await readJson<DatasetV9Config>(configPath);
validateConfig(config);
const groundingLockPath = resolve(projectRoot, argumentValue('--lock') ?? 'configs/v9-grounding-lock.json');
const groundingLockContent = await readFile(groundingLockPath, 'utf8');
const groundingLock = JSON.parse(groundingLockContent) as GroundingLock;
if (groundingLock.config_sha256 !== sha256(configContent)) {
  throw new Error('V9 config changed after the grounding lock.');
}
for (const [path, expected] of Object.entries(groundingLock.implementation)) {
  if (sha256(await readFile(resolve(projectRoot, path), 'utf8')) !== expected) {
    throw new Error(`V9 grounding implementation changed after lock: ${path}`);
  }
}
const sourceManifestPath = resolve(projectRoot, config.sourceManifest);
const sourceManifest = await readJson<Record<string, any>>(sourceManifestPath);
const symbolicLockPath = resolve(projectRoot, config.symbolicLock);
const symbolicAuditPath = resolve(projectRoot, config.symbolicAudit);
const symbolicLockContent = await readFile(symbolicLockPath, 'utf8');
const symbolicAudit = await readJson<Record<string, any>>(symbolicAuditPath);
if (!symbolicAudit.passed || symbolicAudit.decision !== 'authorize_v9_grounding_generation') {
  throw new Error('V9 symbolic oracle audit did not authorize grounding generation.');
}
if (symbolicAudit.protocol_lock_sha256 !== sha256(symbolicLockContent)) {
  throw new Error('V9 symbolic audit does not share the configured lock.');
}
if (groundingLock.symbolic_audit_sha256 !== sha256(await readFile(symbolicAuditPath, 'utf8'))) {
  throw new Error('V9 symbolic audit changed after the grounding lock.');
}

const sourceRecords: V8StructuredRecord[] = [];
for (const configuredPath of config.sourceRecords) {
  const path = resolve(projectRoot, configuredPath);
  const content = await readFile(path, 'utf8');
  const artifactKey = relative(dirname(sourceManifestPath), path);
  if (sourceManifest.artifact_sha256[artifactKey] !== sha256(content)) {
    throw new Error(`V9 source record changed: ${configuredPath}`);
  }
  sourceRecords.push(...content.trim().split('\n').filter(Boolean).map((line) => JSON.parse(line) as V8StructuredRecord));
}
const records = buildV9GroundingRecords(
  sourceRecords,
  sourceManifest.dataset_sha256,
  config.contextCalibrationModulo,
);
const validation = validateV9Grounding(records);
if (validation.errors.length > 0) {
  throw new Error(`V9 validation failed:\n${validation.errors.join('\n')}\n${JSON.stringify(validation, null, 2)}`);
}

const outputDir = resolve(projectRoot, config.outputDir);
await rm(outputDir, { recursive: true, force: true });
const artifactHashes: Record<string, string> = {};
const hashParts: string[] = [];
for (const split of ['train', 'calibration'] as const) {
  const relativePath = `records/${split}.jsonl`;
  const content = await writeJsonl(
    resolve(outputDir, relativePath),
    records.filter((record): record is V9GroundingRecord => record.split === split),
  );
  artifactHashes[relativePath] = sha256(content);
  hashParts.push(`${relativePath}\n${content}`);
}
const implementationPaths = [
  'src/compile-v9.ts',
  'src/v9-grounding.ts',
  'src/v9-validation.ts',
  'src/v9-symbolic.ts',
  'src/v9-contracts.ts',
];
const manifest = {
  schema_version: 9,
  created_at: new Date().toISOString(),
  experiment: 'v9_natural_language_evidence_grounding_development',
  config,
  config_sha256: sha256(configContent),
  source_v8_manifest: config.sourceManifest,
  source_v8_manifest_sha256: sha256(await readFile(sourceManifestPath, 'utf8')),
  source_v8_dataset_sha256: sourceManifest.dataset_sha256,
  symbolic_lock: config.symbolicLock,
  symbolic_lock_sha256: sha256(symbolicLockContent),
  symbolic_audit: config.symbolicAudit,
  symbolic_audit_sha256: sha256(await readFile(symbolicAuditPath, 'utf8')),
  grounding_lock: relative(projectRoot, groundingLockPath),
  grounding_lock_sha256: sha256(groundingLockContent),
  source_simulation_sha256: sourceManifest.source_simulation_sha256,
  source_commit: sourceCommit(),
  implementation_sha256: Object.fromEntries(await Promise.all(implementationPaths.map(async (path) => [
    path,
    sha256(await readFile(resolve(projectRoot, path), 'utf8')),
  ]))),
  artifact_sha256: artifactHashes,
  validation,
  data_access: {
    v3_test_records_read: 0,
    prior_holdout_records_read: 0,
    v7_tone_drift_records_read: 0,
    v7_model_results_read: 0,
    untouched_v8_mechanic_records_read: 0,
    final_v9_mechanic_records_created: 0,
    final_v9_mechanic_model_scores_read: 0,
  },
  dataset_sha256: sha256(hashParts.join('')),
};
await writeJson(resolve(outputDir, 'manifest.json'), manifest);
await writeJson(resolve(outputDir, 'validation.json'), validation);
console.log(JSON.stringify({ outputDir: relative(projectRoot, outputDir), ...manifest }, null, 2));

function validateConfig(value: DatasetV9Config): void {
  if (!value.outputDir.startsWith('data/') || value.outputDir === 'data/') {
    throw new Error('V9 outputDir must be a specific data subdirectory.');
  }
  if (canonicalJson(value.sourceReplicas) !== canonicalJson([0, 1, 2, 3])) {
    throw new Error('V9 requires exactly one source replica per language template.');
  }
  if (canonicalJson([...value.templateFamilies].sort()) !== canonicalJson([...v9TemplateFamilies].sort())) {
    throw new Error('V9 template families differ from the generator.');
  }
  const configuredOperators = Object.fromEntries(Object.entries(value.operatorFamilies).flatMap(([operator, mechanics]) =>
    mechanics.map((mechanic) => [mechanic, operator]),
  ));
  if (canonicalJson(configuredOperators) !== canonicalJson(v9OperatorByMechanic)) {
    throw new Error('V9 operator-family assignment differs from the generator.');
  }
  if (value.contextCalibrationModulo !== 4) throw new Error('V9 context calibration modulo must be 4.');
}

function argumentValue(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function sourceCommit(): string | null {
  try {
    return execFileSync('git', ['rev-parse', 'HEAD'], { cwd: simulatorRoot, encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
}
