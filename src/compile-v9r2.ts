import { readFile, rm } from 'node:fs/promises';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { V9GroundingRecord } from './v9-contracts';
import { removeSyntheticSceneIdentifiers, validateV9r2 } from './v9r2-grounding';
import { readJson, writeJson, writeJsonl } from './io';
import { sha256 } from './serialization';

interface V9r2Config {
  outputDir: string;
  sourceManifest: string;
  sourceRecords: string[];
  failedShortcutAudit: string;
  shortcutGates: Record<string, number>;
  protocol: Record<string, unknown>;
}

interface V9r2Lock {
  config_sha256: string;
  failed_shortcut_audit_sha256: string;
  implementation: Record<string, string>;
}

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const configPath = resolve(projectRoot, argumentValue('--config') ?? 'configs/dataset.v9r2.json');
const lockPath = resolve(projectRoot, argumentValue('--lock') ?? 'configs/v9r2-grounding-lock.json');
const configContent = await readFile(configPath, 'utf8');
const lockContent = await readFile(lockPath, 'utf8');
const config = await readJson<V9r2Config>(configPath);
const lock = JSON.parse(lockContent) as V9r2Lock;
if (lock.config_sha256 !== sha256(configContent)) throw new Error('V9r2 config changed after lock.');
for (const [path, expected] of Object.entries(lock.implementation)) {
  if (sha256(await readFile(resolve(projectRoot, path), 'utf8')) !== expected) {
    throw new Error(`V9r2 implementation changed after lock: ${path}`);
  }
}
const failedAuditPath = resolve(projectRoot, config.failedShortcutAudit);
if (lock.failed_shortcut_audit_sha256 !== sha256(await readFile(failedAuditPath, 'utf8'))) {
  throw new Error('V9 failed shortcut audit changed after V9r2 lock.');
}
const sourceManifestPath = resolve(projectRoot, config.sourceManifest);
const sourceManifest = await readJson<Record<string, any>>(sourceManifestPath);
const sourceRecords: V9GroundingRecord[] = [];
for (const sourceText of config.sourceRecords) {
  const path = resolve(projectRoot, sourceText);
  const content = await readFile(path, 'utf8');
  const key = relative(dirname(sourceManifestPath), path);
  if (sourceManifest.artifact_sha256[key] !== sha256(content)) throw new Error(`V9r2 source changed: ${sourceText}`);
  sourceRecords.push(...content.trim().split('\n').filter(Boolean).map((line) => JSON.parse(line) as V9GroundingRecord));
}
const records = removeSyntheticSceneIdentifiers(sourceRecords);
const validation = validateV9r2(records);
if (validation.errors.length > 0) {
  throw new Error(`V9r2 validation failed:\n${validation.errors.join('\n')}\n${JSON.stringify(validation, null, 2)}`);
}
const outputDir = resolve(projectRoot, config.outputDir);
await rm(outputDir, { recursive: true, force: true });
const artifactHashes: Record<string, string> = {};
const hashParts: string[] = [];
for (const split of ['train', 'calibration'] as const) {
  const relativePath = `records/${split}.jsonl`;
  const content = await writeJsonl(resolve(outputDir, relativePath), records.filter((record) => record.split === split));
  artifactHashes[relativePath] = sha256(content);
  hashParts.push(`${relativePath}\n${content}`);
}
const implementationPaths = ['src/compile-v9r2.ts', 'src/v9r2-grounding.ts'];
const manifest = {
  schema_version: 9,
  revision: 2,
  created_at: new Date().toISOString(),
  experiment: 'v9r2_natural_language_evidence_grounding_development',
  config,
  config_sha256: sha256(configContent),
  grounding_lock: relative(projectRoot, lockPath),
  grounding_lock_sha256: sha256(lockContent),
  source_v9_manifest: config.sourceManifest,
  source_v9_manifest_sha256: sha256(await readFile(sourceManifestPath, 'utf8')),
  source_v9_dataset_sha256: sourceManifest.dataset_sha256,
  failed_shortcut_audit: config.failedShortcutAudit,
  failed_shortcut_audit_sha256: sha256(await readFile(failedAuditPath, 'utf8')),
  source_v8_dataset_sha256: sourceManifest.source_v8_dataset_sha256,
  source_simulation_sha256: sourceManifest.source_simulation_sha256,
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

function argumentValue(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}
