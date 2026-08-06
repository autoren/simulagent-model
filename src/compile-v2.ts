import { mkdir, rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type {
  CounterfactualRecord,
  DatasetManifest,
  DatasetSplit,
  DatasetV2Config,
} from './contracts';
import { readJson, readJsonl, writeJson, writeJsonl } from './io';
import { sha256 } from './serialization';
import {
  buildAgentEpistemicRecords,
  balanceOutcomeCountTraining,
  buildPrivilegedV2Records,
  toAgentV2Mlx,
  toOutcomeCountMlx,
  toPrivilegedV2Mlx,
} from './v2';
import { validateV2 } from './v2-validation';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const configPath = resolve(projectRoot, argumentValue('--config') ?? 'configs/dataset.v2.json');
const config = await readJson<DatasetV2Config>(configPath);
validateConfig(config);
const inputDir = resolve(projectRoot, config.inputDir);
const outputDir = resolve(projectRoot, config.outputDir);
const source: CounterfactualRecord[] = [];
for (const split of ['train', 'valid', 'test'] as const) {
  source.push(...(await readJsonl<CounterfactualRecord>(resolve(inputDir, 'records', `${split}.jsonl`))));
}

const agent = buildAgentEpistemicRecords({
  source,
  splitSeed: config.splitSeed,
  splitRatios: config.splitRatios,
});
const privileged = buildPrivilegedV2Records({
  source,
  splitSeed: config.splitSeed,
  splitRatios: config.splitRatios,
});
const validation = validateV2(agent, privileged);

await rm(outputDir, { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });
const contents: string[] = [];
for (const split of ['train', 'valid', 'test'] as DatasetSplit[]) {
  const agentRecords = agent.filter((record) => record.split === split);
  const privilegedRecords = privileged.filter((record) => record.split === split);
  await writeArtifact(`records/agent/${split}.jsonl`, agentRecords);
  await writeArtifact(`records/privileged/${split}.jsonl`, privilegedRecords);
  await writeArtifact(`mlx/agent/${split}.jsonl`, agentRecords.map(toAgentV2Mlx));
  await writeArtifact(`mlx/outcome-count/${split}.jsonl`, agentRecords.map(toOutcomeCountMlx));
  const balancedCountRecords =
    split === 'train' ? balanceOutcomeCountTraining(agentRecords) : agentRecords;
  await writeArtifact(
    `mlx/outcome-count-balanced/${split}.jsonl`,
    balancedCountRecords.map(toOutcomeCountMlx),
  );
  await writeArtifact(`mlx/privileged/${split}.jsonl`, privilegedRecords.map(toPrivilegedV2Mlx));
}

const sourceManifest = await readJson<DatasetManifest>(resolve(inputDir, 'manifest.json'));
const manifest = {
  schema_version: 2,
  created_at: new Date().toISOString(),
  config,
  source_dataset_sha256: sourceManifest.dataset_sha256,
  source_records: source.length,
  validation,
  dataset_sha256: sha256(contents.join('')),
};
await writeJson(resolve(outputDir, 'manifest.json'), manifest);
await writeJson(resolve(outputDir, 'validation.json'), validation);
console.log(JSON.stringify({ outputDir, ...manifest }, null, 2));

async function writeArtifact(relativePath: string, values: unknown[]): Promise<void> {
  const content = await writeJsonl(resolve(outputDir, relativePath), values);
  contents.push(`${relativePath}\n${content}`);
}

function argumentValue(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function validateConfig(value: DatasetV2Config): void {
  if (!value.inputDir.startsWith('data/') || !value.outputDir.startsWith('data/')) {
    throw new Error('v2 inputDir and outputDir must stay under data/.');
  }
  const total = value.splitRatios.train + value.splitRatios.valid + value.splitRatios.test;
  if (Math.abs(total - 1) > 1e-9) {
    throw new Error('v2 split ratios must sum to one.');
  }
}
