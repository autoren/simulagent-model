import { execFileSync } from 'node:child_process';
import { readFile, rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { DatasetV8Config, V8Mechanic, V8Split } from './contracts';
import { readJson, writeJson, writeJsonl } from './io';
import { sha256 } from './serialization';
import { buildV8Records, v8MechanicOrder } from './v8';
import { validateV8 } from './v8-validation';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const simulatorRoot = resolve(projectRoot, '../simulagent');
const configPath = resolve(projectRoot, argumentValue('--config') ?? 'configs/dataset.v8.json');
const configContent = await readFile(configPath, 'utf8');
const config = await readJson<DatasetV8Config>(configPath);
validateConfig(config);
const outputDir = resolve(projectRoot, config.outputDir);

// V8 is development-only. It reads no Tone Drift record, V7 score, V3 test record,
// or prior holdout. All labels below are recomputed from resolveAction.
const built = buildV8Records(config);
const { records: _records, ...buildSummary } = built;
const validation = validateV8({
  records: built.records,
  mechanics: config.mechanics,
  surfaces: config.surfaceVariants,
  minimumLabelFlipGroupsPerMechanic: 20,
});
if (validation.errors.length > 0) {
  throw new Error(`V8 validation failed:\n${validation.errors.join('\n')}\n${JSON.stringify(validation, null, 2)}`);
}

await rm(outputDir, { recursive: true, force: true });
const artifactHashes: Record<string, string> = {};
const hashParts: string[] = [];
for (const split of ['train', 'calibration'] as V8Split[]) {
  const relativePath = `records/${split}.jsonl`;
  const content = await writeJsonl(
    resolve(outputDir, relativePath),
    built.records.filter((record) => record.split === split),
  );
  artifactHashes[relativePath] = sha256(content);
  hashParts.push(`${relativePath}\n${content}`);
}

const implementationPaths = [
  'src/compile-v8.ts',
  'src/v8.ts',
  'src/v8-validation.ts',
  'src/contracts.ts',
];
const manifest = {
  schema_version: 8,
  created_at: new Date().toISOString(),
  experiment: 'v8_structured_causal_development',
  config,
  config_sha256: sha256(configContent),
  source_project: simulatorRoot,
  source_commit: sourceCommit(),
  source_simulation_sha256: sha256(await readFile(resolve(simulatorRoot, 'src/simulation.ts'), 'utf8')),
  source_worktree_diff_sha256: sourceDiffHash(),
  implementation_sha256: Object.fromEntries(await Promise.all(implementationPaths.map(async (path) => [
    path,
    sha256(await readFile(resolve(projectRoot, path), 'utf8')),
  ]))),
  source_scenarios: built.source_scenarios,
  build: buildSummary,
  artifact_sha256: artifactHashes,
  validation,
  data_access: {
    v3_test_records_read: 0,
    prior_holdout_records_read: 0,
    v7_tone_drift_records_read: 0,
    v7_model_results_read: 0,
    untouched_v8_mechanic_records_created: 0,
    untouched_v8_mechanic_model_scores_read: 0,
  },
  dataset_sha256: sha256(hashParts.join('')),
};
await writeJson(resolve(outputDir, 'manifest.json'), manifest);
await writeJson(resolve(outputDir, 'validation.json'), validation);
console.log(JSON.stringify({ outputDir, ...manifest }, null, 2));

function validateConfig(value: DatasetV8Config): void {
  if (!value.outputDir.startsWith('data/') || value.outputDir === 'data/') {
    throw new Error('V8 outputDir must be a specific directory below data/.');
  }
  if (canonicalStrings(value.mechanics) !== canonicalStrings(v8MechanicOrder)) {
    throw new Error('V8 requires exactly the six preregistered exposed development mechanics.');
  }
  if (canonicalStrings(value.surfaceVariants) !== 'canonical,entity_renamed,paraphrased') {
    throw new Error('V8 requires complete canonical/entity-renamed/paraphrased surfaces.');
  }
  if (value.replicasPerAssignment < 12 || value.calibrationModulo < 3) {
    throw new Error('V8 requires at least 12 replicas and a calibration modulo of at least 3.');
  }
  if (new Set(Object.values(value.simulatorSeeds)).size !== value.mechanics.length) {
    throw new Error('V8 simulator seeds must be unique across mechanics.');
  }
  if (value.protocol.model !== 'mlx-community/Qwen3.5-0.8B-4bit' ||
      value.protocol.feature !== 'layer_06_mean' ||
      value.protocol.cValue !== 10 ||
      value.protocol.seed !== 0) {
    throw new Error('V8 stage 3 freezes Qwen3.5-0.8B layer-6 mean, C=10, seed 0.');
  }
  for (const [name, gate] of Object.entries({ ...value.shortcutGates, ...value.protocol.gates })) {
    if (!(gate > 0 && gate <= 1)) throw new Error(`V8 gate ${name} must be in (0, 1].`);
  }
}

function canonicalStrings(values: readonly string[]): string {
  return [...new Set(values)].sort().join(',');
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

function sourceDiffHash(): string {
  try {
    const diff = execFileSync(
      'git',
      ['diff', '--no-ext-diff', '--', 'src/simulation.ts', 'src/simulation.test.ts', 'src/experiments.ts'],
      { cwd: simulatorRoot, encoding: 'utf8', maxBuffer: 20 * 1024 * 1024 },
    );
    return sha256(diff);
  } catch {
    return sha256('unavailable');
  }
}
