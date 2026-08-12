import { execFileSync } from 'node:child_process';
import { readFile, rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  generateBehavioralTrapScenarioCatalog,
  registerScenarioVariants,
} from '../../simulagent/src/simulation';
import type {
  AgentIdentifiabilityRecordV4,
  DatasetV5ChallengeConfig,
} from './contracts';
import { compileScenario } from './dataset';
import { readJson, readJsonl, writeJson, writeJsonl } from './io';
import { canonicalJson, sha256 } from './serialization';
import { buildAgentEpistemicRecords } from './v2';
import { buildV5ChallengeRecords } from './v5-challenge';
import { validateV5Challenge } from './v5-challenge-validation';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const configPath = resolve(
  projectRoot,
  argumentValue('--config') ?? 'configs/dataset.v5.challenge.json',
);
const config = await readJson<DatasetV5ChallengeConfig>(configPath);
validateConfig(config);
const outputDir = resolve(projectRoot, config.outputDir);
const developmentDir = resolve(projectRoot, config.sourceDevelopmentDir);
const lockPath = resolve(projectRoot, config.frozenProbeLock);

const developmentRecords = (
  await Promise.all(
    ['train', 'calibration', 'validation'].map((split) =>
      readJsonl<AgentIdentifiabilityRecordV4>(resolve(developmentDir, `${split}.jsonl`)),
    ),
  )
).flat();
const developmentPromptKeys = new Set(
  developmentRecords.map((record) => canonicalJson(record.agent_input)),
);
const batches = config.scenarioSeeds.map((seed) => {
  const scenarios = registerScenarioVariants(
    generateBehavioralTrapScenarioCatalog({
      seeds: [seed],
      behavioralMechanics: config.mechanics,
      behavioralVariants: config.evidenceVariants,
    }),
  );
  const source = scenarios.flatMap((scenario) =>
    compileScenario({
      scenario,
      split: 'valid',
      maxStates: config.maxStatesPerScenario,
      maxDepth: config.maxDepth,
    }),
  );
  const aggregated = buildAgentEpistemicRecords({
    source,
    splitSeed: `v5-challenge-aggregation-only:${seed}`,
    splitRatios: { train: 0, valid: 1, test: 0 },
  });
  return { scenarios, source, aggregated };
});
const scenarios = batches.flatMap((batch) => batch.scenarios);
const source = batches.flatMap((batch) => batch.source);
const aggregated = batches.flatMap((batch) => batch.aggregated);
const built = buildV5ChallengeRecords({
  records: aggregated,
  developmentPromptKeys,
  surfaceVariants: config.surfaceVariants,
});
const validation = validateV5Challenge({
  records: built.records,
  developmentRecords,
  requiredSurfaces: config.surfaceVariants,
});
if (validation.errors.length > 0) {
  throw new Error(`V5 challenge validation failed:\n${validation.errors.join('\n')}`);
}

await rm(outputDir, { recursive: true, force: true });
const recordsContent = await writeJsonl(resolve(outputDir, 'records/challenge.jsonl'), built.records);
const lockContent = await readFile(lockPath, 'utf8');
const manifest = {
  schema_version: 5,
  created_at: new Date().toISOString(),
  config,
  source_project: resolve(projectRoot, '../simulagent'),
  source_commit: sourceCommit(),
  source_scenarios: scenarios.map((scenario) => scenario.id).sort(),
  source_counterfactual_records: source.length,
  aggregated_agent_records: aggregated.length,
  selected_base_records: built.base_records,
  excluded_development_prompt_overlaps: built.excluded_development_prompt_overlaps,
  frozen_probe_lock_sha256: sha256(lockContent),
  validation,
  source_test_records_read: 0,
  dataset_sha256: sha256(`records/challenge.jsonl\n${recordsContent}`),
};
await writeJson(resolve(outputDir, 'manifest.json'), manifest);
await writeJson(resolve(outputDir, 'validation.json'), validation);
console.log(JSON.stringify({ outputDir, ...manifest }, null, 2));

function argumentValue(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function validateConfig(value: DatasetV5ChallengeConfig): void {
  if (!value.outputDir.startsWith('data/') || value.outputDir === 'data/') {
    throw new Error('V5 challenge outputDir must be a specific directory below data/.');
  }
  if (!value.sourceDevelopmentDir.startsWith('data/v4/records')) {
    throw new Error('V5 challenge may compare overlap only against V4 development records.');
  }
  if (value.scenarioSeeds.length < 3 || new Set(value.scenarioSeeds).size !== value.scenarioSeeds.length) {
    throw new Error('V5 challenge requires at least three unique scenario seeds.');
  }
  if (value.maxStatesPerScenario < 1 || value.maxDepth < 0) {
    throw new Error('V5 challenge state/depth bounds are invalid.');
  }
  if (!value.mechanics.includes('powertrip') || !value.mechanics.includes('relockshort')) {
    throw new Error('V5 challenge requires both powertrip and relockshort mechanics.');
  }
  const requiredSurfaces = ['canonical', 'entity_renamed', 'paraphrased'];
  if (requiredSurfaces.some((surface) => !value.surfaceVariants.includes(surface as never))) {
    throw new Error('V5 challenge requires canonical, entity-renamed, and paraphrased surfaces.');
  }
  for (const [name, threshold] of Object.entries(value.evaluationGates)) {
    if (!(threshold > 0.5 && threshold <= 1)) {
      throw new Error(`V5 challenge gate ${name} must be above 0.5 and at most 1.`);
    }
  }
}

function sourceCommit(): string | null {
  try {
    return execFileSync('git', ['rev-parse', 'HEAD'], {
      cwd: resolve(projectRoot, '../simulagent'),
      encoding: 'utf8',
    }).trim();
  } catch {
    return null;
  }
}
