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
  AgentIdentifiabilityRecordV4,
  DatasetV6Config,
  V5ChallengeRecord,
  V6Mechanic,
  V6Split,
} from './contracts';
import { compileScenario } from './dataset';
import { readJson, readJsonl, writeJson, writeJsonl } from './io';
import { sha256 } from './serialization';
import { buildAgentEpistemicRecords } from './v2';
import { buildV6Records, mergeObservationallyEquivalentRecords } from './v6';
import { validateV6 } from './v6-validation';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const configPath = resolve(projectRoot, argumentValue('--config') ?? 'configs/dataset.v6.json');
const configContent = await readFile(configPath, 'utf8');
const config = JSON.parse(configContent) as DatasetV6Config;
validateConfig(config);
const outputDir = resolve(projectRoot, config.outputDir);

// The audit reads only V4 development and the already consumed V5 challenge. V3 test stays closed.
const priorDevelopmentRecords = (
  await Promise.all(
    ['train', 'calibration', 'validation'].map((split) =>
      readJsonl<AgentIdentifiabilityRecordV4>(
        resolve(projectRoot, config.priorDevelopmentDir, `${split}.jsonl`),
      ),
    ),
  )
).flat();
const priorChallengeRecords = await readJsonl<V5ChallengeRecord>(
  resolve(projectRoot, config.priorChallengeRecords),
);

const development = compilePartition(
  config.developmentSeeds,
  config.developmentMechanics,
  'development',
);
const holdout = compilePartition(
  config.holdoutSeeds,
  [config.holdoutMechanic],
  'mechanic-holdout',
);
const built = buildV6Records({
  developmentRecords: development.merged,
  holdoutRecords: holdout.merged,
  surfaceVariants: config.surfaceVariants,
  calibrationRatio: config.calibrationRatio,
  splitSeed: 'v6-connected-context-stratification',
  stratificationRestarts: config.stratificationRestarts,
});
const validation = validateV6({
  records: built.records,
  priorDevelopmentRecords,
  priorChallengeRecords,
  requiredSurfaces: config.surfaceVariants,
  developmentMechanics: config.developmentMechanics,
  holdoutMechanic: config.holdoutMechanic,
  developmentSeeds: config.developmentSeeds,
  holdoutSeeds: config.holdoutSeeds,
  minimumEvidenceInterventionGroups: config.minimumEvidenceInterventionGroups,
  maximumAmbiguityRateGap: config.maximumAmbiguityRateGap,
});
if (validation.errors.length > 0) {
  throw new Error(
    `V6 validation failed:\n${validation.errors.join('\n')}\n${JSON.stringify(validation, null, 2)}`,
  );
}

await rm(outputDir, { recursive: true, force: true });
const artifactHashes: Record<string, string> = {};
const hashParts: string[] = [];
for (const split of ['train', 'calibration', 'mechanic_holdout'] as V6Split[]) {
  const relativePath = `records/${split}.jsonl`;
  const content = await writeJsonl(
    resolve(outputDir, relativePath),
    built.records.filter((record) => record.split === split),
  );
  artifactHashes[relativePath] = sha256(content);
  hashParts.push(`${relativePath}\n${content}`);
}
const manifest = {
  schema_version: 6,
  created_at: new Date().toISOString(),
  config,
  config_sha256: sha256(configContent),
  source_project: resolve(projectRoot, '../simulagent'),
  source_commit: sourceCommit(),
  source_simulation_sha256: sha256(
    await readFile(resolve(projectRoot, '../simulagent/src/simulation.ts'), 'utf8'),
  ),
  source_worktree_diff_sha256: sourceDiffHash(),
  source_scenarios: {
    development: development.scenarios.map((scenario) => scenario.id).sort(),
    mechanic_holdout: holdout.scenarios.map((scenario) => scenario.id).sort(),
  },
  source_counterfactual_records: {
    development: development.sourceRecords,
    mechanic_holdout: holdout.sourceRecords,
  },
  aggregated_agent_records: {
    development: development.aggregatedRecords,
    mechanic_holdout: holdout.aggregatedRecords,
  },
  merged_agent_records: {
    development: development.merged.length,
    mechanic_holdout: holdout.merged.length,
  },
  build: {
    base_records: built.base_records,
    evidence_intervention_groups: built.evidence_intervention_groups,
    evidence_intervention_groups_by_mechanic: built.evidence_intervention_groups_by_mechanic,
    stratification: built.stratification,
  },
  prior_artifacts: {
    v4_manifest_sha256: sha256(await readFile(resolve(projectRoot, 'data/v4/manifest.json'), 'utf8')),
    v5_challenge_manifest_sha256: sha256(
      await readFile(resolve(projectRoot, 'data/v5-challenge/manifest.json'), 'utf8'),
    ),
  },
  artifact_sha256: artifactHashes,
  validation,
  source_test_records_read: 0,
  dataset_sha256: sha256(hashParts.join('')),
};
await writeJson(resolve(outputDir, 'manifest.json'), manifest);
await writeJson(resolve(outputDir, 'validation.json'), validation);
console.log(JSON.stringify({ outputDir, ...manifest }, null, 2));

function compilePartition(
  seeds: number[],
  mechanics: V6Mechanic[],
  label: string,
): {
  scenarios: ScenarioConfig[];
  sourceRecords: number;
  aggregatedRecords: number;
  merged: AgentEpistemicRecord[];
} {
  const scenarios: ScenarioConfig[] = [];
  let sourceRecords = 0;
  let aggregatedRecords = 0;
  const mergedByMechanic: AgentEpistemicRecord[] = [];
  for (const mechanic of mechanics) {
    const mechanicRecords: AgentEpistemicRecord[] = [];
    for (const seed of seeds) {
      const generated = registerScenarioVariants(
        generateBehavioralTrapScenarioCatalog({
          seeds: [seed],
          behavioralMechanics: [mechanic as BehavioralTrapMechanic],
          behavioralVariants: config.evidenceVariants,
        }),
      );
      scenarios.push(...generated);
      const source = generated.flatMap((scenario) =>
        compileScenario({
          scenario,
          split: 'train',
          maxStates: config.maxStatesPerScenario,
          maxDepth: config.maxDepth,
        }),
      );
      const aggregated = buildAgentEpistemicRecords({
        source,
        splitSeed: `v6-${label}-aggregation-only:${mechanic}:${seed}`,
        splitRatios: { train: 1, valid: 0, test: 0 },
      });
      sourceRecords += source.length;
      aggregatedRecords += aggregated.length;
      mechanicRecords.push(...aggregated);
    }
    mergedByMechanic.push(...mergeObservationallyEquivalentRecords(mechanicRecords));
  }
  return { scenarios, sourceRecords, aggregatedRecords, merged: mergedByMechanic };
}

function validateConfig(value: DatasetV6Config): void {
  if (!value.outputDir.startsWith('data/') || value.outputDir === 'data/') {
    throw new Error('V6 outputDir must be a specific directory below data/.');
  }
  if (value.priorDevelopmentDir !== 'data/v4/records') {
    throw new Error('V6 leakage audit must use only the V4 development record directory.');
  }
  if (value.priorChallengeRecords !== 'data/v5-challenge/records/challenge.jsonl') {
    throw new Error('V6 leakage audit must include the frozen V5 challenge records.');
  }
  if (new Set(value.developmentSeeds).size !== value.developmentSeeds.length || value.developmentSeeds.length < 4) {
    throw new Error('V6 requires at least four unique development seeds.');
  }
  if (new Set(value.holdoutSeeds).size !== value.holdoutSeeds.length || value.holdoutSeeds.length < 3) {
    throw new Error('V6 requires at least three unique holdout seeds.');
  }
  if (value.developmentSeeds.some((seed) => value.holdoutSeeds.includes(seed))) {
    throw new Error('V6 development and holdout seeds must be disjoint.');
  }
  if (canonicalMechanics(value.developmentMechanics) !== 'powertrip,relockshort') {
    throw new Error('V6 development must contain exactly relockshort and powertrip.');
  }
  if (value.holdoutMechanic !== 'mirrorreject') {
    throw new Error('V6 must reserve mirrorreject as the mechanic holdout.');
  }
  if (!(value.calibrationRatio > 0 && value.calibrationRatio < 0.5)) {
    throw new Error('V6 calibrationRatio must be between zero and 0.5.');
  }
  if (value.maxStatesPerScenario < 1 || value.maxDepth < 0) {
    throw new Error('V6 state/depth bounds are invalid.');
  }
  if (!Number.isInteger(value.stratificationRestarts) || value.stratificationRestarts < 1) {
    throw new Error('V6 stratificationRestarts must be positive.');
  }
  const requiredSurfaces = ['canonical', 'entity_renamed', 'paraphrased'];
  if (requiredSurfaces.some((surface) => !value.surfaceVariants.includes(surface as V5SurfaceVariantNever))) {
    throw new Error('V6 requires canonical, entity-renamed, and paraphrased surfaces.');
  }
  if (value.protocol.feature !== 'layer_06_mean' || value.protocol.cValue !== 10 || value.protocol.seed !== 0) {
    throw new Error('V6 baseline must keep the preregistered layer-6 mean, C=10, seed-0 method.');
  }
  for (const [name, gate] of Object.entries(value.protocol.gates)) {
    if (!(gate > 0 && gate <= 1)) throw new Error(`V6 gate ${name} must be in (0, 1].`);
  }
}

type V5SurfaceVariantNever = DatasetV6Config['surfaceVariants'][number];

function canonicalMechanics(values: string[]): string {
  return [...new Set(values)].sort().join(',');
}

function argumentValue(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
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

function sourceDiffHash(): string {
  try {
    const diff = execFileSync(
      'git',
      ['diff', '--no-ext-diff', '--', 'src/simulation.ts', 'src/behavioralTrapAnalysis.ts', 'src/experiments.ts'],
      { cwd: resolve(projectRoot, '../simulagent'), encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 },
    );
    return sha256(diff);
  } catch {
    return sha256('unavailable');
  }
}
