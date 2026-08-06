import { resolve } from 'node:path';
import type { AgentEpistemicRecordV3, DatasetSplit, DatasetV3Config } from './contracts';
import { readJson, readJsonl } from './io';
import { validateV3 } from './v3-validation';

const projectRoot = resolve(import.meta.dirname, '..');
const config = await readJson<DatasetV3Config>(
  resolve(projectRoot, argumentValue('--config') ?? 'configs/dataset.v3.json'),
);
const records: AgentEpistemicRecordV3[] = [];
for (const split of ['train', 'valid', 'test'] as DatasetSplit[]) {
  records.push(
    ...(await readJsonl<AgentEpistemicRecordV3>(
      resolve(projectRoot, config.outputDir, 'records', 'agent', `${split}.jsonl`),
    )),
  );
}
const report = validateV3(records, {
  maximumAmbiguityRateGap: config.maximumAmbiguityRateGap,
  maximumMechanicShareGap: config.maximumMechanicShareGap,
});
console.log(JSON.stringify(report, null, 2));
if (report.errors.length > 0) process.exitCode = 1;

function argumentValue(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}
