import { access, mkdir, readFile } from 'node:fs/promises';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { V8StructuredRecord } from './contracts';
import { readJson, writeJson, writeJsonl } from './io';
import { canonicalJson, sha256 } from './serialization';
import { buildV17FinalMechanic, type V17ExpectedTopology } from './v17-final-mechanic';

interface V17Config {
  outputDir: string;
  sourceReplica: number;
  sourceV8Manifest: string;
  sourceV8Records: string[];
  sourceV14Manifest: string;
  expected: V17ExpectedTopology;
}

interface V17ConstructionLock {
  config_sha256: string;
  implementation: Record<string, string>;
  source: {
    v8_manifest: string;
    v8_manifest_sha256: string;
    v8_artifacts: Record<string, string>;
    v14_manifest: string;
    v14_manifest_sha256: string;
    v16_result: string;
    v16_result_sha256: string;
  };
}

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const configPath = resolve(projectRoot, argumentValue('--config') ?? 'configs/v17-final-mechanic.json');
const lockPath = resolve(projectRoot, argumentValue('--lock') ?? 'configs/v17-final-construction-lock.json');
const configContent = await readFile(configPath, 'utf8');
const lockContent = await readFile(lockPath, 'utf8');
const config = await readJson<V17Config>(configPath);
const lock = JSON.parse(lockContent) as V17ConstructionLock;
if (lock.config_sha256 !== sha256(configContent)) throw new Error('V17 config changed after construction lock.');
for (const [path, expected] of Object.entries(lock.implementation)) {
  if (sha256(await readFile(resolve(projectRoot, path), 'utf8')) !== expected) throw new Error(`V17 locked implementation changed: ${path}`);
}
const v16Result = await readJson<Record<string, any>>(resolve(projectRoot, lock.source.v16_result));
if (sha256(await readFile(resolve(projectRoot, lock.source.v16_result), 'utf8')) !== lock.source.v16_result_sha256 ||
    v16Result.decision !== 'authorize_separately_locked_final_mechanic_evaluation' || v16Result.final_mechanic_accessed) {
  throw new Error('V16 does not authorize V17 construction.');
}
const v8ManifestPath = resolve(projectRoot, config.sourceV8Manifest);
const v8ManifestContent = await readFile(v8ManifestPath, 'utf8');
if (sha256(v8ManifestContent) !== lock.source.v8_manifest_sha256) throw new Error('V17 V8 manifest changed after lock.');
const v8Manifest = JSON.parse(v8ManifestContent) as Record<string, any>;
const sourceRecords: V8StructuredRecord[] = [];
for (const configured of config.sourceV8Records) {
  const path = resolve(projectRoot, configured);
  const content = await readFile(path, 'utf8');
  if (sha256(content) !== lock.source.v8_artifacts[configured]) throw new Error(`V17 scaffold source changed: ${configured}`);
  sourceRecords.push(...content.trim().split('\n').filter(Boolean).map((line) => JSON.parse(line) as V8StructuredRecord));
}
const v14ManifestPath = resolve(projectRoot, config.sourceV14Manifest);
if (sha256(await readFile(v14ManifestPath, 'utf8')) !== lock.source.v14_manifest_sha256) throw new Error('V17 V14 manifest changed after lock.');
const v14Manifest = await readJson<Record<string, any>>(v14ManifestPath);
const developmentCandidateActions = new Set<string>();
for (const [artifact, expected] of Object.entries(v14Manifest.artifact_sha256 as Record<string, string>)) {
  const path = resolve(dirname(v14ManifestPath), artifact);
  const content = await readFile(path, 'utf8');
  if (sha256(content) !== expected) throw new Error(`V17 V14 development artifact changed: ${artifact}`);
  for (const line of content.trim().split('\n').filter(Boolean)) {
    developmentCandidateActions.add((JSON.parse(line) as any).agent_input.candidate_action as string);
  }
}
const outputDir = resolve(projectRoot, config.outputDir);
try {
  await access(outputDir);
  throw new Error(`V17 final directory already exists; refusing reconstruction: ${config.outputDir}`);
} catch (error) {
  if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
}
const built = buildV17FinalMechanic(
  sourceRecords,
  v8Manifest.dataset_sha256,
  config.expected,
  developmentCandidateActions,
  config.sourceReplica,
);
if (built.validation.errors.length) throw new Error(`V17 final validation failed:\n${built.validation.errors.join('\n')}`);
await mkdir(outputDir, { recursive: false });
const relativeArtifact = 'records/final_mechanic.jsonl';
const content = await writeJsonl(resolve(outputDir, relativeArtifact), built.records);
const manifest = {
  schema_version: 17,
  created_at: new Date().toISOString(),
  experiment: 'v17r2_one_shot_final_mechanic',
  construction_lock: relative(projectRoot, lockPath),
  construction_lock_sha256: sha256(lockContent),
  config_path: relative(projectRoot, configPath),
  config_sha256: sha256(configContent),
  source_v8_manifest: config.sourceV8Manifest,
  source_v8_manifest_sha256: sha256(v8ManifestContent),
  source_v8_dataset_sha256: v8Manifest.dataset_sha256,
  source_v14_manifest: config.sourceV14Manifest,
  source_v14_manifest_sha256: lock.source.v14_manifest_sha256,
  source_v16_result: lock.source.v16_result,
  source_v16_result_sha256: lock.source.v16_result_sha256,
  source_scaffolds: built.source_scaffolds,
  transition_table_sha256: built.transition_table_sha256,
  artifact_sha256: { [relativeArtifact]: sha256(content) },
  validation: built.validation,
  data_access: {
    final_v17_mechanic_records_created: built.records.length,
    final_v17_model_scores_read: 0,
    v3_test_records_read: 0,
    prior_holdout_records_read: 0,
    v7_tone_drift_records_read: 0,
    v7_model_results_read: 0,
    untouched_v8_mechanic_records_read: 0,
  },
  dataset_sha256: sha256(`${relativeArtifact}\n${content}`),
};
await writeJson(resolve(outputDir, 'manifest.json'), manifest);
await writeJson(resolve(outputDir, 'validation.json'), built.validation);
console.log(canonicalJson({ outputDir: config.outputDir, ...manifest }));

function argumentValue(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}
