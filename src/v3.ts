import type {
  AgentEpistemicRecordV3,
  CounterfactualRecord,
  DatasetSplit,
  SplitRatios,
} from './contracts';
import { canonicalJson } from './serialization';
import { createStratifiedSplitPlan, type StratifiedGroup } from './stratified-split';
import { buildAgentEpistemicRecords } from './v2';

export interface V3BuildResult {
  records: AgentEpistemicRecordV3[];
  stratification: {
    objective: number;
    restarts: number;
    eligible_mechanic_labels: string[];
  };
}

export function buildAgentEpistemicRecordsV3(options: {
  source: CounterfactualRecord[];
  splitSeed: string;
  splitRatios: SplitRatios;
  stratificationRestarts: number;
  minimumMechanicSupport: number;
}): V3BuildResult {
  const scenarioLabels = scenarioMechanicLabels(options.source);
  const base = buildAgentEpistemicRecords({
    source: options.source,
    splitSeed: `${options.splitSeed}:aggregation-only`,
    splitRatios: options.splitRatios,
  }).map(
    (record): AgentEpistemicRecordV3 => ({
      ...record,
      id: record.id.replace('agent-v2:', 'agent-v3:'),
      schema_version: 3,
      mechanic_labels: [
        ...new Set(
          record.source_scenario_ids.flatMap((scenarioId) => scenarioLabels.get(scenarioId) ?? []),
        ),
      ].sort(),
    }),
  );
  const mechanicSupport = new Map<string, number>();
  for (const record of base) {
    for (const label of record.mechanic_labels) {
      mechanicSupport.set(label, (mechanicSupport.get(label) ?? 0) + 1);
    }
  }
  const eligibleMechanicLabels = [...mechanicSupport]
    .filter(([label, support]) => label.startsWith('family:') || support >= options.minimumMechanicSupport)
    .map(([label]) => label)
    .sort();
  const eligible = new Set(eligibleMechanicLabels);
  const byGroup = new Map<string, AgentEpistemicRecordV3[]>();
  for (const record of base) {
    byGroup.set(record.split_group, [...(byGroup.get(record.split_group) ?? []), record]);
  }
  const groups: StratifiedGroup[] = [...byGroup].map(([id, records]) => ({
    id,
    features: groupFeatures(records, eligible),
  }));
  const stratification = createStratifiedSplitPlan(
    groups,
    options.splitRatios,
    options.splitSeed,
    options.stratificationRestarts,
  );
  const records = base
    .map((record) => ({ ...record, split: requiredSplit(stratification.plan, record.split_group) }))
    .sort(recordOrder);
  return {
    records,
    stratification: {
      objective: stratification.objective,
      restarts: stratification.restarts,
      eligible_mechanic_labels: eligibleMechanicLabels,
    },
  };
}

function scenarioMechanicLabels(source: CounterfactualRecord[]): Map<string, string[]> {
  const labels = new Map<string, string[]>();
  for (const record of source) {
    labels.set(
      record.scenario_id,
      [`family:${record.scenario_family}`, ...record.scenario_tags.map((tag) => `tag:${tag}`)].sort(),
    );
  }
  return labels;
}

function groupFeatures(
  records: AgentEpistemicRecordV3[],
  eligibleMechanics: Set<string>,
): Record<string, number> {
  const features: Record<string, number> = { records: records.length };
  for (const record of records) {
    increment(features, `class:${record.target.identifiable ? 'identifiable' : 'ambiguous'}`);
    increment(features, `count:${record.target.possible_outcomes.length}`);
    increment(features, `action:${record.agent_input.candidate_action.key.split(':', 1)[0]}`);
    for (const label of record.mechanic_labels) {
      if (eligibleMechanics.has(label)) {
        increment(features, label);
      }
    }
  }
  return features;
}

function increment(values: Record<string, number>, key: string): void {
  values[key] = (values[key] ?? 0) + 1;
}

function requiredSplit(plan: Map<string, DatasetSplit>, group: string): DatasetSplit {
  const split = plan.get(group);
  if (!split) throw new Error(`Missing v3 split for ${group}.`);
  return split;
}

function recordOrder(
  left: { split: DatasetSplit; split_group: string; id: string },
  right: { split: DatasetSplit; split_group: string; id: string },
): number {
  return canonicalJson([left.split, left.split_group, left.id]).localeCompare(
    canonicalJson([right.split, right.split_group, right.id]),
  );
}
