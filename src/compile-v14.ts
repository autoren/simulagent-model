import { readFile, rm } from 'node:fs/promises';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { V8StructuredRecord } from './contracts';
import type { V9OperatorFamily } from './v9-contracts';
import type { V14SemanticOperator, V14SurfaceFamily } from './v14-contracts';
import { buildV14GroundingRecords, v14SurfaceFamilies, v14SurfacesByOperator } from './v14-grounding';
import { validateV14Grounding } from './v14-validation';
import { v10OperatorByMechanic } from './v10-grounding';
import { readJson, writeJson, writeJsonl } from './io';
import { canonicalJson, sha256 } from './serialization';

interface V14Config {
  outputDir: string;
  sourceManifest: string;
  sourceRecords: string[];
  symbolicLock: string;
  symbolicAudit: string;
  sourceReplica: number;
  semanticOperators: Record<V14SemanticOperator, V14SurfaceFamily[]>;
  operatorFamilies: Record<V9OperatorFamily, string[]>;
  shortcutGates: Record<string, number>;
}

interface V14Lock {
  config_sha256: string;
  symbolic_audit_sha256: string;
  source_manifest_sha256: string;
  source_artifact_sha256: Record<string, string>;
  implementation: Record<string, string>;
}

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const configPath = resolve(projectRoot, argumentValue('--config') ?? 'configs/dataset.v14.json');
const lockPath = resolve(projectRoot, argumentValue('--lock') ?? 'configs/v14-grounding-lock.json');
const configContent = await readFile(configPath, 'utf8');
const lockContent = await readFile(lockPath, 'utf8');
const config = await readJson<V14Config>(configPath);
const lock = JSON.parse(lockContent) as V14Lock;
validateConfig(config);
if (lock.config_sha256 !== sha256(configContent)) throw new Error('V14 config changed after lock.');
for (const [path, expected] of Object.entries(lock.implementation)) {
  if (sha256(await readFile(resolve(projectRoot, path), 'utf8')) !== expected) {
    throw new Error(`V14 implementation changed after lock: ${path}`);
  }
}
const sourceManifestPath = resolve(projectRoot, config.sourceManifest);
if (lock.source_manifest_sha256 !== sha256(await readFile(sourceManifestPath, 'utf8'))) {
  throw new Error('V14 source manifest changed after lock.');
}
const sourceManifest = await readJson<Record<string, any>>(sourceManifestPath);
const symbolicAuditPath = resolve(projectRoot, config.symbolicAudit);
const symbolicAudit = await readJson<Record<string, any>>(symbolicAuditPath);
if (!symbolicAudit.passed || symbolicAudit.decision !== 'authorize_v9_grounding_generation') {
  throw new Error('V9 symbolic audit does not authorize V14 generation.');
}
if (lock.symbolic_audit_sha256 !== sha256(await readFile(symbolicAuditPath, 'utf8'))) {
  throw new Error('V14 symbolic audit changed after lock.');
}

const sourceRecords: V8StructuredRecord[] = [];
for (const configuredPath of config.sourceRecords) {
  const path = resolve(projectRoot, configuredPath);
  const content = await readFile(path, 'utf8');
  const artifactKey = relative(dirname(sourceManifestPath), path);
  if (sourceManifest.artifact_sha256[artifactKey] !== sha256(content)) throw new Error(`V14 source changed: ${configuredPath}`);
  if (lock.source_artifact_sha256[configuredPath] !== sha256(content)) throw new Error(`V14 lock source changed: ${configuredPath}`);
  sourceRecords.push(...content.trim().split('\n').filter(Boolean).map((line) => JSON.parse(line) as V8StructuredRecord));
}
const records = buildV14GroundingRecords(sourceRecords, sourceManifest.dataset_sha256, config.sourceReplica);
const validation = validateV14Grounding(records);
if (validation.errors.length) throw new Error(`V14 validation failed:\n${validation.errors.join('\n')}`);

const outputDir = resolve(projectRoot, config.outputDir);
await rm(outputDir, { recursive: true, force: true });
const artifactHashes: Record<string, string> = {};
const hashParts: string[] = [];
for (const split of ['train', 'evaluation'] as const) {
  const relativePath = `records/${split}.jsonl`;
  const content = await writeJsonl(resolve(outputDir, relativePath), records.filter((record) => record.split === split));
  artifactHashes[relativePath] = sha256(content);
  hashParts.push(`${relativePath}\n${content}`);
}
const manifest = {
  schema_version: 14,
  created_at: new Date().toISOString(),
  experiment: 'v14_operator_supported_grounding_development',
  config,
  config_sha256: sha256(configContent),
  grounding_lock: relative(projectRoot, lockPath),
  grounding_lock_sha256: sha256(lockContent),
  source_v8_manifest: config.sourceManifest,
  source_v8_manifest_sha256: sha256(await readFile(sourceManifestPath, 'utf8')),
  source_v8_dataset_sha256: sourceManifest.dataset_sha256,
  source_simulation_sha256: sourceManifest.source_simulation_sha256,
  symbolic_audit: config.symbolicAudit,
  symbolic_audit_sha256: sha256(await readFile(symbolicAuditPath, 'utf8')),
  implementation_sha256: lock.implementation,
  artifact_sha256: artifactHashes,
  validation,
  data_access: {
    v3_test_records_read: 0,
    prior_holdout_records_read: 0,
    v7_tone_drift_records_read: 0,
    v7_model_results_read: 0,
    untouched_v8_mechanic_records_read: 0,
    final_v9_mechanic_records_read: 0,
  },
  dataset_sha256: sha256(hashParts.join('')),
};
await writeJson(resolve(outputDir, 'manifest.json'), manifest);
await writeJson(resolve(outputDir, 'validation.json'), validation);
console.log(JSON.stringify({ outputDir: relative(projectRoot, outputDir), ...manifest }, null, 2));

function validateConfig(value: V14Config): void {
  if (!value.outputDir.startsWith('data/') || value.outputDir === 'data/') throw new Error('V14 outputDir must be specific.');
  if (value.sourceReplica !== 0) throw new Error('V14 source replica must remain zero.');
  if (canonicalJson(value.semanticOperators) !== canonicalJson(v14SurfacesByOperator)) {
    throw new Error('V14 semantic operator assignment differs from the generator.');
  }
  if (canonicalJson(Object.values(value.semanticOperators).flat().sort()) !== canonicalJson([...v14SurfaceFamilies].sort())) {
    throw new Error('V14 surface family list differs from the generator.');
  }
  const configured = Object.fromEntries(Object.entries(value.operatorFamilies).flatMap(([operator, mechanics]) =>
    mechanics.map((mechanic) => [mechanic, operator]),
  ));
  if (canonicalJson(configured) !== canonicalJson(v10OperatorByMechanic)) throw new Error('V14 transition operators differ.');
}

function argumentValue(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}
