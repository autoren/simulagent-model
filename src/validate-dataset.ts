import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import type { CounterfactualRecord, DatasetSplit } from './contracts';
import { validateRecords } from './validation';

const projectRoot = resolve(import.meta.dirname, '..');
const datasetDir = resolve(projectRoot, process.argv[2] ?? 'data/pilot');
const records: CounterfactualRecord[] = [];

for (const split of ['train', 'valid', 'test'] as DatasetSplit[]) {
  const path = resolve(datasetDir, 'records', `${split}.jsonl`);
  const content = await readFile(path, 'utf8');
  for (const line of content.split('\n').filter(Boolean)) {
    records.push(JSON.parse(line) as CounterfactualRecord);
  }
}

console.log(JSON.stringify(validateRecords(records), null, 2));

