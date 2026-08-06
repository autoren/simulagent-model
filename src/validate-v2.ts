import { resolve } from 'node:path';
import type {
  AgentEpistemicRecord,
  DatasetSplit,
  PrivilegedTransitionRecordV2,
} from './contracts';
import { readJsonl } from './io';
import { validateV2 } from './v2-validation';

const datasetDir = resolve(process.argv[2] ?? 'data/v2');
const agent: AgentEpistemicRecord[] = [];
const privileged: PrivilegedTransitionRecordV2[] = [];
for (const split of ['train', 'valid', 'test'] as DatasetSplit[]) {
  agent.push(
    ...(await readJsonl<AgentEpistemicRecord>(
      resolve(datasetDir, 'records', 'agent', `${split}.jsonl`),
    )),
  );
  privileged.push(
    ...(await readJsonl<PrivilegedTransitionRecordV2>(
      resolve(datasetDir, 'records', 'privileged', `${split}.jsonl`),
    )),
  );
}
console.log(JSON.stringify(validateV2(agent, privileged), null, 2));
