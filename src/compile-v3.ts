import { mkdir, rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type {
  CounterfactualRecord,
  DatasetManifest,
  DatasetSplit,
  DatasetV3Config,
} from './contracts';
import { readJson, readJsonl, writeJson, writeJsonl } from './io';
import { sha256 } from './serialization';
import { balanceOutcomeCountTraining, toAgentV2Mlx, toOutcomeCountMlx } from './v2';
import { buildAgentEpistemicRecordsV3 } from './v3';
import { validateV3 } from './v3-validation';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const configPath = resolve(projectRoot, argumentValue('--config') ?? 'configs/dataset.v3.json');
const config = await readJson<DatasetV3Config>(configPath);
validateConfig(config);
const inputDir = resolve(projectRoot, config.inputDir);
const outputDir = resolve(projectRoot, config.outputDir);
const source: CounterfactualRecord[] = [];
for (const split of ['train', 'valid', 'test'] as const) {
  source.push(...(await readJsonl<CounterfactualRecord>(resolve(inputDir, 'records', `${split}.jsonl`))));
}

const built = buildAgentEpistemicRecordsV3({
  source,
  splitSeed: config.splitSeed,
  splitRatios: config.splitRatios,
  stratificationRestarts: config.stratificationRestarts,
  minimumMechanicSupport: config.minimumMechanicSupport,
});
const validation = validateV3(built.records, {
  maximumAmbiguityRateGap: config.maximumAmbiguityRateGap,
  maximumMechanicShareGap: config.maximumMechanicShareGap,
});
if (validation.errors.length > 0) {
  throw new Error(`V3 validation failed:\n${validation.errors.join('\n')}`);
}

await rm(outputDir, { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });
const contents: string[] = [];
for (const split of ['train', 'valid', 'test'] as DatasetSplit[]) {
  const records = built.records.filter((record) => record.split === split);
  await writeArtifact(`records/agent/${split}.jsonl`, records);
  await writeArtifact(`mlx/agent/${split}.jsonl`, records.map(toAgentV2Mlx));
  await writeArtifact(`mlx/outcome-count/${split}.jsonl`, records.map(toOutcomeCountMlx));
  const balanced = split === 'train' ? balanceOutcomeCountTraining(records) : records;
  await writeArtifact(`mlx/outcome-count-balanced/${split}.jsonl`, balanced.map(toOutcomeCountMlx));
}

const sourceManifest = await readJson<DatasetManifest>(resolve(inputDir, 'manifest.json'));
const manifest = {
  schema_version: 3,
  created_at: new Date().toISOString(),
  config,
  source_dataset_sha256: sourceManifest.dataset_sha256,
  source_records: source.length,
  stratification: built.stratification,
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

function validateConfig(value: DatasetV3Config): void {
  if (!value.inputDir.startsWith('data/') || !value.outputDir.startsWith('data/')) {
    throw new Error('V3 inputDir and outputDir must stay under data/.');
  }
  const total = value.splitRatios.train + value.splitRatios.valid + value.splitRatios.test;
  if (Math.abs(total - 1) > 1e-9) throw new Error('V3 split ratios must sum to one.');
  if (value.stratificationRestarts < 1) throw new Error('V3 requires stratification restarts.');
  if (value.minimumMechanicSupport < 1) throw new Error('V3 mechanic support must be positive.');
}
