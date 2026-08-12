import { execFileSync } from 'node:child_process';
import { readFile, rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  generateBehavioralTrapScenarioCatalog,
  registerScenarioVariants,
  type BehavioralTrapMechanic,
  type ScenarioConfig,
} from '../../simulagent/src/simulation';
import type {
  AgentEpistemicRecord,
  DatasetV7Config,
  V5EvidenceVariant,
  V7DevelopmentMechanic,
  V7Split,
} from './contracts';
import { compileScenario } from './dataset';
import { readJson, writeJson, writeJsonl } from './io';
import { sha256 } from './serialization';
import { buildAgentEpistemicRecords } from './v2';
import { evidenceVariant } from './v5-challenge';
import { mergeObservationallyEquivalentRecords } from './v6';
import { buildV7Records, type V7SourceCandidate } from './v7';
import { validateV7 } from './v7-validation';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const simulatorRoot = resolve(projectRoot, '../simulagent');
const configPath = resolve(projectRoot, argumentValue('--config') ?? 'configs/dataset.v7.json');
const configContent = await readFile(configPath, 'utf8');
const config = await readJson<DatasetV7Config>(configPath);
validateConfig(config);
const outputDir = resolve(projectRoot, config.outputDir);

// V7 intentionally reads no V3 record and no V5/V6 evaluation or holdout artifact.
const development = compileDevelopmentCandidates(config);
const holdout = compileHoldoutCandidates(config);
const built = buildV7Records({
  developmentCandidates: development.candidates,
  holdoutCandidates: holdout.candidates,
  evidenceVariants: config.evidenceVariants,
  surfaceVariants: config.surfaceVariants,
  calibrationRatio: config.calibrationRatio,
  stratificationRestarts: config.stratificationRestarts,
  maximumPairsPerConditionalStratum: config.maximumPairsPerConditionalStratum,
  minimumLabelChangingDevelopmentGroups: config.minimumLabelChangingDevelopmentGroups,
});
const validation = validateV7({
  records: built.records,
  requiredSurfaces: config.surfaceVariants,
  requiredEvidenceVariants: config.evidenceVariants,
  developmentMechanics: config.developmentMechanics,
  holdoutMechanic: config.holdoutMechanic,
  developmentSeeds: config.developmentSeeds,
  holdoutSeeds: config.holdoutSeeds,
  minimumLabelChangingDevelopmentGroups: config.minimumLabelChangingDevelopmentGroups,
  maximumConditionalLabelGap: config.maximumConditionalLabelGap,
});
if (validation.errors.length > 0) {
  throw new Error(`V7 validation failed:\n${validation.errors.join('\n')}\n${JSON.stringify(validation, null, 2)}`);
}

await rm(outputDir, { recursive: true, force: true });
const artifactHashes: Record<string, string> = {};
const hashParts: string[] = [];
for (const split of ['train', 'calibration', 'untouched_mechanic'] as V7Split[]) {
  const relativePath = `records/${split}.jsonl`;
  const content = await writeJsonl(
    resolve(outputDir, relativePath),
    built.records.filter((record) => record.split === split),
  );
  artifactHashes[relativePath] = sha256(content);
  hashParts.push(`${relativePath}\n${content}`);
}

const simulatorSourcePath = resolve(simulatorRoot, 'src/simulation.ts');
const manifest = {
  schema_version: 7,
  created_at: new Date().toISOString(),
  config,
  config_sha256: sha256(configContent),
  source_project: simulatorRoot,
  source_commit: sourceCommit(),
  source_simulation_sha256: sha256(await readFile(simulatorSourcePath, 'utf8')),
  source_worktree_diff_sha256: sourceDiffHash(),
  source_scenarios: {
    development: development.scenarios.map((scenario) => scenario.id).sort(),
    untouched_mechanic: holdout.scenarios.map((scenario) => scenario.id).sort(),
  },
  source_counterfactual_records: {
    development: development.sourceRecords,
    untouched_mechanic: holdout.sourceRecords,
  },
  aggregated_agent_records: {
    development: development.aggregatedRecords,
    untouched_mechanic: holdout.aggregatedRecords,
  },
  build: built,
  artifact_sha256: artifactHashes,
  validation,
  data_access: {
    v3_test_records_read: 0,
    prior_holdout_records_read: 0,
    prior_model_results_read: 0,
    untouched_mechanic_model_scores_read: 0,
  },
  dataset_sha256: sha256(hashParts.join('')),
};
await writeJson(resolve(outputDir, 'manifest.json'), manifest);
await writeJson(resolve(outputDir, 'validation.json'), validation);
console.log(JSON.stringify({ outputDir, ...manifest }, null, 2));

function compileDevelopmentCandidates(config: DatasetV7Config): {
  scenarios: ScenarioConfig[];
  sourceRecords: number;
  aggregatedRecords: number;
  candidates: V7SourceCandidate[];
} {
  const scenarios: ScenarioConfig[] = [];
  const candidates: V7SourceCandidate[] = [];
  let sourceRecords = 0;
  let aggregatedRecords = 0;
  for (const mechanic of config.developmentMechanics) {
    for (const evidence of config.evidenceVariants) {
      const records: AgentEpistemicRecord[] = [];
      for (const seed of config.developmentSeeds) {
        const compiled = compileScenarioPair(config, mechanic, evidence, seed);
        scenarios.push(...compiled.scenarios);
        sourceRecords += compiled.sourceRecords;
        aggregatedRecords += compiled.aggregated.length;
        records.push(...compiled.aggregated);
      }
      for (const record of mergeObservationallyEquivalentRecords(records)) {
        candidates.push({
          record,
          mechanic,
          evidence,
          seeds: sourceSeeds(record.source_scenario_ids, mechanic),
        });
      }
    }
  }
  return { scenarios, sourceRecords, aggregatedRecords, candidates };
}

function compileHoldoutCandidates(config: DatasetV7Config): {
  scenarios: ScenarioConfig[];
  sourceRecords: number;
  aggregatedRecords: number;
  candidates: V7SourceCandidate[];
} {
  const scenarios: ScenarioConfig[] = [];
  const records: AgentEpistemicRecord[] = [];
  let sourceRecords = 0;
  let aggregatedRecords = 0;
  for (const seed of config.holdoutSeeds) {
    const generated = registerScenarioVariants(generateBehavioralTrapScenarioCatalog({
      seeds: [seed],
      behavioralMechanics: [config.holdoutMechanic],
      behavioralVariants: config.evidenceVariants,
    }));
    scenarios.push(...generated);
    const source = generated.flatMap((scenario) => compileScenario({
      scenario,
      split: 'train',
      maxStates: config.maxStatesPerScenario,
      maxDepth: config.maxDepth,
    }));
    const aggregated = buildAgentEpistemicRecords({
      source,
      splitSeed: `v7-untouched-aggregation:${config.holdoutMechanic}:${seed}`,
      splitRatios: { train: 1, valid: 0, test: 0 },
    });
    sourceRecords += source.length;
    aggregatedRecords += aggregated.length;
    records.push(...aggregated);
  }
  const candidates = mergeObservationallyEquivalentRecords(records).map((record): V7SourceCandidate => ({
    record,
    mechanic: config.holdoutMechanic,
    evidence: evidenceVariant(record.source_scenario_ids),
    seeds: sourceSeeds(record.source_scenario_ids, config.holdoutMechanic),
  }));
  return { scenarios, sourceRecords, aggregatedRecords, candidates };
}

function compileScenarioPair(
  config: DatasetV7Config,
  mechanic: V7DevelopmentMechanic,
  evidence: Exclude<V5EvidenceVariant, 'mixed'>,
  seed: number,
): { scenarios: ScenarioConfig[]; sourceRecords: number; aggregated: AgentEpistemicRecord[] } {
  const scenarios = registerScenarioVariants(generateBehavioralTrapScenarioCatalog({
    seeds: [seed],
    behavioralMechanics: [mechanic as BehavioralTrapMechanic],
    behavioralVariants: [evidence],
  }));
  const source = scenarios.flatMap((scenario) => compileScenario({
    scenario,
    split: 'train',
    maxStates: config.maxStatesPerScenario,
    maxDepth: config.maxDepth,
  }));
  const aggregated = buildAgentEpistemicRecords({
    source,
    splitSeed: `v7-development-aggregation:${mechanic}:${evidence}:${seed}`,
    splitRatios: { train: 1, valid: 0, test: 0 },
  });
  return { scenarios, sourceRecords: source.length, aggregated };
}

function sourceSeeds(sourceIds: string[], mechanic: string): number[] {
  const seeds = new Set<number>();
  for (const id of sourceIds) {
    const match = id.match(new RegExp(`-${mechanic}-(\\d+)-(?:trap|control)$`));
    if (!match) throw new Error(`Invalid V7 ${mechanic} source scenario ${id}.`);
    seeds.add(Number(match[1]));
  }
  return [...seeds].sort((left, right) => left - right);
}

function validateConfig(value: DatasetV7Config): void {
  if (!value.outputDir.startsWith('data/') || value.outputDir === 'data/') {
    throw new Error('V7 outputDir must be a specific directory below data/.');
  }
  if (canonicalStrings(value.developmentMechanics) !== 'powertrip,relockshort') {
    throw new Error('V7 development must use exactly fresh relockshort and powertrip scenarios.');
  }
  if (value.holdoutMechanic !== 'tonedrift') {
    throw new Error('V7 must reserve the new tone-drift mechanic.');
  }
  if (new Set(value.developmentSeeds).size !== value.developmentSeeds.length || value.developmentSeeds.length < 6) {
    throw new Error('V7 requires at least six unique development seeds.');
  }
  if (new Set(value.holdoutSeeds).size !== value.holdoutSeeds.length || value.holdoutSeeds.length < 4) {
    throw new Error('V7 requires at least four unique holdout seeds.');
  }
  if (value.developmentSeeds.some((seed) => value.holdoutSeeds.includes(seed))) {
    throw new Error('V7 development and untouched seeds must be disjoint.');
  }
  if (canonicalStrings(value.evidenceVariants) !== canonicalStrings([
    'forced',
    'announced',
    'announced-upstream',
    'announced-consequence',
    'announced-procedure',
    'unobservable',
  ])) {
    throw new Error('V7 requires every preregistered evidence rung exactly once.');
  }
  if (canonicalStrings(value.surfaceVariants) !== 'canonical,entity_renamed,paraphrased') {
    throw new Error('V7 requires complete canonical/entity-renamed/paraphrased groups.');
  }
  if (!(value.calibrationRatio > 0 && value.calibrationRatio < 0.5)) {
    throw new Error('V7 calibrationRatio must be between zero and 0.5.');
  }
  if (value.maximumPairsPerConditionalStratum < 1 || value.minimumLabelChangingDevelopmentGroups < 2) {
    throw new Error('V7 curriculum support bounds are invalid.');
  }
  if (!(value.maximumConditionalLabelGap >= 0 && value.maximumConditionalLabelGap <= 0.1)) {
    throw new Error('V7 conditional label gap must be at most 0.1.');
  }
  if (value.protocol.feature !== 'layer_06_mean' || value.protocol.cValue !== 10 || value.protocol.seed !== 0) {
    throw new Error('V7 must freeze the 0.8B layer-6 mean, C=10, seed-0 baseline.');
  }
  for (const [name, gate] of Object.entries({ ...value.shortcutGates, ...value.protocol.gates })) {
    if (!(gate > 0 && gate <= 1)) throw new Error(`V7 gate ${name} must be in (0, 1].`);
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
