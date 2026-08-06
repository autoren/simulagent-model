import { resolve } from 'node:path';
import type { AgentIdentifiabilityRecordV4, DatasetV4Config } from './contracts';
import { readJson, readJsonl } from './io';
import { validateV4 } from './v4-validation';

const config = await readJson<DatasetV4Config>(resolve('configs/dataset.v4.json'));
const records: AgentIdentifiabilityRecordV4[] = [];
for (const split of ['train', 'calibration', 'validation']) {
  records.push(
    ...(await readJsonl<AgentIdentifiabilityRecordV4>(
      resolve(config.outputDir, 'records', `${split}.jsonl`),
    )),
  );
}
const report = validateV4(records, {
  maximumAmbiguityRateGap: config.maximumAmbiguityRateGap,
  maximumMechanicShareGap: config.maximumMechanicShareGap,
});
console.log(JSON.stringify(report, null, 2));
if (report.errors.length > 0) process.exitCode = 1;
