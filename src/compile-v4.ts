import { mkdir, rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type {
  AgentEpistemicRecordV3,
  AgentIdentifiabilityRecordV4,
  DatasetV4Config,
  V4DevelopmentSplit,
} from './contracts';
import { readJson, readJsonl, writeJson, writeJsonl } from './io';
import { sha256 } from './serialization';
import { balanceBinaryTraining, buildBinaryIdentifiabilityRecordsV4, toBinaryIdentifiabilityMlx } from './v4';
import { validateV4 } from './v4-validation';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const configPath = resolve(projectRoot, argumentValue('--config') ?? 'configs/dataset.v4.json');
const config = await readJson<DatasetV4Config>(configPath);
validateConfig(config);
const inputDir = resolve(projectRoot, config.inputDir);
const outputDir = resolve(projectRoot, config.outputDir);

// Deliberately read only V3 train and validation. V3 test stays closed.
const sourceTrain = await readJsonl<AgentEpistemicRecordV3>(
  resolve(inputDir, 'records', 'agent', 'train.jsonl'),
);
const sourceValidation = await readJsonl<AgentEpistemicRecordV3>(
  resolve(inputDir, 'records', 'agent', 'valid.jsonl'),
);
const built = buildBinaryIdentifiabilityRecordsV4({
  sourceTrain,
  sourceValidation,
  splitSeed: config.splitSeed,
  calibrationRatio: config.calibrationRatio,
  stratificationRestarts: config.stratificationRestarts,
});
const validation = validateV4(built.records, {
  maximumAmbiguityRateGap: config.maximumAmbiguityRateGap,
  maximumMechanicShareGap: config.maximumMechanicShareGap,
});
if (validation.errors.length > 0) {
  throw new Error(`V4 validation failed:\n${validation.errors.join('\n')}`);
}

await rm(outputDir, { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });
const contents: string[] = [];
for (const split of ['train', 'calibration', 'validation'] as V4DevelopmentSplit[]) {
  const records = built.records.filter((record) => record.split === split);
  await writeArtifact(`records/${split}.jsonl`, records);
}
const train = built.records.filter((record) => record.split === 'train');
const calibration = built.records.filter((record) => record.split === 'calibration');
const heldValidation = built.records.filter((record) => record.split === 'validation');
await writeArtifact('mlx/binary/train.jsonl', balanceBinaryTraining(train).map(toBinaryIdentifiabilityMlx));
await writeArtifact('mlx/binary/valid.jsonl', calibration.map(toBinaryIdentifiabilityMlx));
await writeArtifact('mlx/binary/validation.jsonl', heldValidation.map(toBinaryIdentifiabilityMlx));

const sourceManifest = await readJson<{ dataset_sha256: string }>(resolve(inputDir, 'manifest.json'));
const manifest = {
  schema_version: 4,
  created_at: new Date().toISOString(),
  config,
  source_dataset_sha256: sourceManifest.dataset_sha256,
  source_train_records: sourceTrain.length,
  source_validation_records: sourceValidation.length,
  source_test_records_read: 0,
  stratification: built.stratification,
  validation,
  label_mapping: { A: 'identifiable', B: 'ambiguous' },
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

function validateConfig(value: DatasetV4Config): void {
  if (!value.inputDir.startsWith('data/') || !value.outputDir.startsWith('data/')) {
    throw new Error('V4 inputDir and outputDir must stay under data/.');
  }
  if (!(value.calibrationRatio > 0 && value.calibrationRatio < 0.5)) {
    throw new Error('V4 calibrationRatio must be greater than zero and less than 0.5.');
  }
  if (!Number.isInteger(value.stratificationRestarts) || value.stratificationRestarts < 1) {
    throw new Error('V4 stratificationRestarts must be a positive integer.');
  }
}
