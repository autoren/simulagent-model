import { execFileSync } from 'node:child_process';
import { mkdir, rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  scenarioVariantList,
  scenarioVariants,
  type ScenarioConfig,
  type ScenarioVariantId,
} from '../../simulagent/src/simulation';
import type {
  CounterfactualRecord,
  DatasetConfig,
  DatasetManifest,
  DatasetSplit,
} from './contracts';
import { compileScenario, recordSortKey } from './dataset';
import { readJson, writeJson, writeJsonl } from './io';
import { toMlxExample } from './mlx';
import { sha256 } from './serialization';
import { createSplitPlan, splitGroupForScenario } from './split';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const configPath = resolve(projectRoot, argumentValue('--config') ?? 'configs/dataset.pilot.json');
const config = await readJson<DatasetConfig>(configPath);
validateConfig(config);

const scenarios = resolveScenarios(config.scenarioIds);
const groups = scenarios.map((scenario) => splitGroupForScenario(scenario.id));
const splitPlan = createSplitPlan(groups, config.splitRatios, config.splitSeed);
const outputDir = resolve(projectRoot, config.outputDir);
await rm(outputDir, { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });

const records = scenarios
  .flatMap((scenario) => {
    const group = splitGroupForScenario(scenario.id);
    const split = splitPlan.get(group);
    if (!split) {
      throw new Error(`No split assigned to ${group}.`);
    }
    return compileScenario({
      scenario,
      split,
      maxStates: config.maxStatesPerScenario,
      maxDepth: config.maxDepth,
    });
  })
  .sort((left, right) => recordSortKey(left).localeCompare(recordSortKey(right)));

const splitOrder: DatasetSplit[] = ['train', 'valid', 'test'];
const datasetContent: string[] = [];
for (const split of splitOrder) {
  const splitRecords = records.filter((record) => record.split === split);
  const recordsPath = `records/${split}.jsonl`;
  const recordsContent = await writeJsonl(resolve(outputDir, recordsPath), splitRecords);
  datasetContent.push(`${recordsPath}\n${recordsContent}`);
  for (const track of ['agent', 'privileged'] as const) {
    const mlxPath = `mlx/${track}/${split}.jsonl`;
    const mlxContent = await writeJsonl(
      resolve(outputDir, mlxPath),
      splitRecords.map((record) => toMlxExample(record, track)),
    );
    datasetContent.push(`${mlxPath}\n${mlxContent}`);
  }
}

const manifest: DatasetManifest = {
  schema_version: 1,
  created_at: new Date().toISOString(),
  source_project: resolve(projectRoot, '../simulagent'),
  source_commit: sourceCommit(),
  config,
  counts: countBySplit(records),
  group_counts: countGroupsBySplit(records),
  scenario_counts: Object.fromEntries(
    scenarios.map((scenario) => [
      scenario.id,
      records.filter((record) => record.scenario_id === scenario.id).length,
    ]),
  ),
  dataset_sha256: sha256(datasetContent.join('')),
};
await writeJson(resolve(outputDir, 'manifest.json'), manifest);

console.log(JSON.stringify({ outputDir, ...manifest }, null, 2));

function argumentValue(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function resolveScenarios(ids: string[] | '*'): ScenarioConfig[] {
  if (ids === '*') {
    return [...scenarioVariantList].sort((left, right) => left.id.localeCompare(right.id));
  }
  return ids.map((id) => {
    const scenario = scenarioVariants[id as ScenarioVariantId];
    if (!scenario) {
      throw new Error(`Unknown Simulagent scenario: ${id}`);
    }
    return scenario;
  });
}

function validateConfig(value: DatasetConfig): void {
  if (value.maxStatesPerScenario < 1 || value.maxDepth < 0) {
    throw new Error('maxStatesPerScenario must be positive and maxDepth cannot be negative.');
  }
  if (!value.outputDir.startsWith('data/')) {
    throw new Error('outputDir must stay under the project data directory.');
  }
}

function countBySplit(records: CounterfactualRecord[]): Record<DatasetSplit, number> {
  return {
    train: records.filter((record) => record.split === 'train').length,
    valid: records.filter((record) => record.split === 'valid').length,
    test: records.filter((record) => record.split === 'test').length,
  };
}

function countGroupsBySplit(records: CounterfactualRecord[]): Record<DatasetSplit, number> {
  return Object.fromEntries(
    splitOrder.map((split) => [
      split,
      new Set(
        records.filter((record) => record.split === split).map((record) => record.split_group),
      ).size,
    ]),
  ) as Record<DatasetSplit, number>;
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
