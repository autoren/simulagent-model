import type {
  AgentEpistemicRecordV3,
  AgentIdentifiabilityRecordV4,
  MlxExample,
  V4DevelopmentSplit,
} from './contracts';
import { canonicalJson } from './serialization';
import { createStratifiedSplitPlan, type StratifiedGroup } from './stratified-split';

export const binaryIdentifiabilitySystemPrompt = [
  'Classify whether the candidate action has exactly one transition supported by the observation history.',
  'Use only the supplied observation history and candidate action.',
  'Return exactly one uppercase ASCII letter and nothing else.',
  'Return A when exactly one transition is supported (identifiable).',
  'Return B when multiple transitions are supported (ambiguous).',
].join(' ');

export interface V4BuildResult {
  records: AgentIdentifiabilityRecordV4[];
  stratification: {
    objective: number;
    restarts: number;
    calibration_ratio: number;
  };
}

export function buildBinaryIdentifiabilityRecordsV4(options: {
  sourceTrain: AgentEpistemicRecordV3[];
  sourceValidation: AgentEpistemicRecordV3[];
  splitSeed: string;
  calibrationRatio: number;
  stratificationRestarts: number;
}): V4BuildResult {
  if (!(options.calibrationRatio > 0 && options.calibrationRatio < 0.5)) {
    throw new Error('V4 calibration ratio must be greater than zero and less than 0.5.');
  }
  if (options.sourceTrain.some((record) => record.split !== 'train')) {
    throw new Error('V4 sourceTrain may contain only V3 training records.');
  }
  if (options.sourceValidation.some((record) => record.split !== 'valid')) {
    throw new Error('V4 sourceValidation may contain only V3 validation records.');
  }

  const groups = groupRecords(options.sourceTrain).map(([id, records]): StratifiedGroup => ({
    id,
    features: groupFeatures(records),
  }));
  const stratification = createStratifiedSplitPlan(
    groups,
    { train: 1 - options.calibrationRatio, valid: options.calibrationRatio, test: 0 },
    options.splitSeed,
    options.stratificationRestarts,
  );
  const development = options.sourceTrain.map((record) =>
    toV4Record(record, stratification.plan.get(record.split_group) === 'valid' ? 'calibration' : 'train'),
  );
  const validation = options.sourceValidation.map((record) => toV4Record(record, 'validation'));
  return {
    records: [...development, ...validation].sort(recordOrder),
    stratification: {
      objective: stratification.objective,
      restarts: stratification.restarts,
      calibration_ratio: options.calibrationRatio,
    },
  };
}

export function toBinaryIdentifiabilityMlx(record: AgentIdentifiabilityRecordV4): MlxExample {
  const input = { ...record.agent_input, task: 'classify_identifiability' };
  return {
    messages: [
      { role: 'system', content: binaryIdentifiabilitySystemPrompt },
      { role: 'user', content: canonicalJson(input) },
      { role: 'assistant', content: record.target.identifiable ? 'A' : 'B' },
    ],
  };
}

export function balanceBinaryTraining(
  records: AgentIdentifiabilityRecordV4[],
): AgentIdentifiabilityRecordV4[] {
  const identifiable = records.filter((record) => record.target.identifiable);
  const ambiguous = records.filter((record) => !record.target.identifiable);
  if (identifiable.length === 0 || ambiguous.length === 0) return [...records];
  const targetSize = Math.max(identifiable.length, ambiguous.length);
  const repeatTo = (values: AgentIdentifiabilityRecordV4[]) =>
    Array.from({ length: targetSize }, (_, index) => values[index % values.length]);
  return [...repeatTo(identifiable), ...repeatTo(ambiguous)];
}

function groupRecords(
  records: AgentEpistemicRecordV3[],
): Array<[string, AgentEpistemicRecordV3[]]> {
  const groups = new Map<string, AgentEpistemicRecordV3[]>();
  for (const record of records) {
    groups.set(record.split_group, [...(groups.get(record.split_group) ?? []), record]);
  }
  return [...groups];
}

function groupFeatures(records: AgentEpistemicRecordV3[]): Record<string, number> {
  const features: Record<string, number> = { records: records.length };
  for (const record of records) {
    increment(features, `class:${record.target.identifiable ? 'identifiable' : 'ambiguous'}`);
    increment(features, `count:${record.target.possible_outcomes.length}`);
    increment(features, `action:${record.agent_input.candidate_action.key.split(':', 1)[0]}`);
    for (const label of record.mechanic_labels) increment(features, label);
  }
  return features;
}

function increment(values: Record<string, number>, key: string): void {
  values[key] = (values[key] ?? 0) + 1;
}

function toV4Record(
  record: AgentEpistemicRecordV3,
  split: V4DevelopmentSplit,
): AgentIdentifiabilityRecordV4 {
  return {
    ...record,
    id: record.id.replace('agent-v3:', 'identifiability-v4:'),
    schema_version: 4,
    source_split: record.split,
    split,
  };
}

function recordOrder(
  left: AgentIdentifiabilityRecordV4,
  right: AgentIdentifiabilityRecordV4,
): number {
  return canonicalJson([left.split, left.split_group, left.id]).localeCompare(
    canonicalJson([right.split, right.split_group, right.id]),
  );
}
