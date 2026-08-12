import type {
  AgentEpistemicInput,
  AgentEpistemicRecord,
  V5EvidenceVariant,
  V5SurfaceVariant,
  V6IdentifiabilityRecord,
  V6Mechanic,
  V6Split,
} from './contracts';
import { canonicalJson, shortHash } from './serialization';
import { createStratifiedSplitPlan, type StratifiedGroup } from './stratified-split';
import { evidenceVariant, transformSurface } from './v5-challenge';

interface DescribedRecord {
  record: AgentEpistemicRecord;
  mechanic: V6Mechanic;
  seeds: number[];
}

export interface V6BuildResult {
  records: V6IdentifiabilityRecord[];
  base_records: Record<V6Split, number>;
  evidence_intervention_groups: Record<V6Split, number>;
  evidence_intervention_groups_by_mechanic: Record<V6Split, Partial<Record<V6Mechanic, number>>>;
  stratification: {
    objective: number;
    restarts: number;
    calibration_ratio: number;
    connected_components: number;
  };
}

export function mergeObservationallyEquivalentRecords(
  records: AgentEpistemicRecord[],
): AgentEpistemicRecord[] {
  const byPrompt = groupBy(records, (record) => canonicalJson(record.agent_input));
  return [...byPrompt.entries()].map(([prompt, values]): AgentEpistemicRecord => {
    const outcomes = new Map<string, AgentEpistemicRecord['target']['possible_outcomes'][number]>();
    for (const record of values) {
      for (const outcome of record.target.possible_outcomes) {
        outcomes.set(canonicalJson(outcome), outcome);
      }
    }
    const possibleOutcomes = [...outcomes.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([, value]) => value);
    return {
      id: `agent-v6-source:${shortHash(prompt, 24)}`,
      schema_version: 2,
      split: 'train',
      split_group: values[0].split_group,
      agent_input: values[0].agent_input,
      target: {
        identifiable: possibleOutcomes.length === 1,
        possible_outcomes: possibleOutcomes,
      },
      empirical_support: [],
      source_record_count: values.reduce((sum, value) => sum + value.source_record_count, 0),
      source_scenario_ids: [...new Set(values.flatMap((value) => value.source_scenario_ids))].sort(),
    };
  }).sort(baseRecordOrder);
}

export function buildV6Records(options: {
  developmentRecords: AgentEpistemicRecord[];
  holdoutRecords: AgentEpistemicRecord[];
  surfaceVariants: V5SurfaceVariant[];
  calibrationRatio: number;
  splitSeed: string;
  stratificationRestarts: number;
}): V6BuildResult {
  const development = describe(options.developmentRecords);
  const holdout = describe(options.holdoutRecords);
  if (development.some((value) => value.mechanic === 'mirrorreject')) {
    throw new Error('V6 development records may not contain the held-out mirrorreject mechanic.');
  }
  if (holdout.some((value) => value.mechanic !== 'mirrorreject')) {
    throw new Error('V6 holdout records may contain only the mirrorreject mechanic.');
  }

  const developmentInterventions = evidenceInterventionIds(development);
  const holdoutInterventions = evidenceInterventionIds(holdout);
  const selectedDevelopment = selectBalanced(development, developmentInterventions);
  const selectedHoldout = selectBalanced(holdout, holdoutInterventions);
  const components = connectedComponents(selectedDevelopment, developmentInterventions);
  const groups: StratifiedGroup[] = [...components.entries()].map(([id, values]) => ({
    id,
    features: componentFeatures(values, developmentInterventions),
  }));
  const stratified = createStratifiedSplitPlan(
    groups,
    { train: 1 - options.calibrationRatio, valid: options.calibrationRatio, test: 0 },
    options.splitSeed,
    options.stratificationRestarts,
  );
  const componentForRecord = new Map<string, string>();
  for (const [component, values] of components) {
    for (const value of values) componentForRecord.set(value.record.id, component);
  }
  const developmentSplits = new Map<string, V6Split>();
  for (const value of selectedDevelopment) {
    const component = required(componentForRecord, value.record.id, 'component');
    developmentSplits.set(
      value.record.id,
      required(stratified.plan, component, 'split') === 'valid' ? 'calibration' : 'train',
    );
  }

  const base = [
    ...selectedDevelopment.map((value) => ({
      value,
      split: required(developmentSplits, value.record.id, 'development split'),
      interventionId: developmentInterventions.get(value.record.id) ?? null,
    })),
    ...selectedHoldout.map((value) => ({
      value,
      split: 'mechanic_holdout' as const,
      interventionId: holdoutInterventions.get(value.record.id) ?? null,
    })),
  ];
  const records = base.flatMap(({ value, split, interventionId }) =>
    toSurfaceRecords(value, split, interventionId, options.surfaceVariants),
  ).sort(recordOrder);
  const baseCounts = countBaseBySplit(records);
  const interventionCounts = countInterventionsBySplit(records);
  const interventionByMechanic = countInterventionsByMechanic(records);
  return {
    records,
    base_records: baseCounts,
    evidence_intervention_groups: interventionCounts,
    evidence_intervention_groups_by_mechanic: interventionByMechanic,
    stratification: {
      objective: stratified.objective,
      restarts: stratified.restarts,
      calibration_ratio: options.calibrationRatio,
      connected_components: components.size,
    },
  };
}

export function v6EvidenceSignature(value: DescribedRecord): string {
  const observation = value.record.agent_input.observation;
  return canonicalJson({
    mechanic: value.mechanic,
    turn: observation.turn,
    location: observation.location,
    exits: observation.exits.map((exit) => [exit.direction, exit.blocked]),
    visible_objects: observation.visibleObjects.map((object) => object.id),
    characters: observation.characters.map((character) => character.id),
    inventory: observation.inventory.map((object) => object.id),
    pressure: observation.pressure,
    signal: observation.signal,
    candidate_action: value.record.agent_input.candidate_action.key,
    available_actions: value.record.agent_input.available_actions.map((action) => action.key),
  });
}

function evidenceInterventionIds(records: DescribedRecord[]): Map<string, string> {
  const result = new Map<string, string>();
  const groups = groupBy(records, v6EvidenceSignature);
  for (const [signature, values] of groups) {
    const variants = new Set(
      values.map((value) => evidenceVariant(value.record.source_scenario_ids)).filter((value) => value !== 'mixed'),
    );
    if (variants.size < 2) continue;
    const id = `v6-evidence:${shortHash(signature, 24)}`;
    for (const value of values) result.set(value.record.id, id);
  }
  return result;
}

function selectBalanced(
  records: DescribedRecord[],
  interventionIds: Map<string, string>,
): DescribedRecord[] {
  const selected = new Map<string, DescribedRecord>();
  const byContext = groupBy(records, (value) => value.record.split_group);
  for (const values of byContext.values()) {
    const ambiguous = values.filter((value) => !value.record.target.identifiable).sort(describedOrder);
    const identifiable = values.filter((value) => value.record.target.identifiable).sort(describedOrder);
    if (ambiguous.length === 0 || identifiable.length === 0) continue;
    const pairCount = Math.min(ambiguous.length, identifiable.length);
    ambiguous.slice(0, pairCount).forEach((value) => selected.set(value.record.id, value));
    identifiable.slice(0, pairCount).forEach((value) => selected.set(value.record.id, value));
  }

  const interventionRecords = records.filter((value) => interventionIds.has(value.record.id));
  for (const values of groupBy(
    interventionRecords,
    (value) => required(interventionIds, value.record.id, 'intervention'),
  ).values()) {
    const ordered = [...values].sort((left, right) => {
      const variant = evidenceVariant(left.record.source_scenario_ids).localeCompare(
        evidenceVariant(right.record.source_scenario_ids),
      );
      return variant || describedOrder(left, right);
    });
    const explicit = ordered.filter((value) => evidenceVariant(value.record.source_scenario_ids) !== 'mixed');
    const byVariant = groupBy(explicit, (value) => evidenceVariant(value.record.source_scenario_ids));
    [...byVariant.values()].slice(0, 2).forEach((valuesForVariant) => {
      selected.set(valuesForVariant[0].record.id, valuesForVariant[0]);
    });
    const labels = new Set(values.map((value) => value.record.target.identifiable));
    if (labels.size > 1) {
      for (const identifiable of [true, false]) {
        const match = ordered.find((value) => value.record.target.identifiable === identifiable);
        if (match) selected.set(match.record.id, match);
      }
    }
  }
  rebalanceByMechanic(selected, records, new Set(interventionIds.keys()));
  return [...selected.values()].sort(describedOrder);
}

function rebalanceByMechanic(
  selected: Map<string, DescribedRecord>,
  candidates: DescribedRecord[],
  protectedIds: Set<string>,
): void {
  const mechanics = new Set(candidates.map((value) => value.mechanic));
  for (const mechanic of mechanics) {
    const selectedForMechanic = () => [...selected.values()].filter((value) => value.mechanic === mechanic);
    let values = selectedForMechanic();
    const identifiableCount = values.filter((value) => value.record.target.identifiable).length;
    const ambiguousCount = values.length - identifiableCount;
    const minorityIdentifiable = identifiableCount < ambiguousCount;
    let needed = Math.abs(identifiableCount - ambiguousCount);
    const additions = candidates
      .filter((value) =>
        value.mechanic === mechanic &&
        !selected.has(value.record.id) &&
        value.record.target.identifiable === minorityIdentifiable,
      )
      .sort(describedOrder);
    for (const value of additions.slice(0, needed)) selected.set(value.record.id, value);
    values = selectedForMechanic();
    const newIdentifiable = values.filter((value) => value.record.target.identifiable).length;
    const newAmbiguous = values.length - newIdentifiable;
    needed = Math.abs(newIdentifiable - newAmbiguous);
    if (needed === 0) continue;
    const majorityIdentifiable = newIdentifiable > newAmbiguous;
    const removable = values
      .filter((value) =>
        value.record.target.identifiable === majorityIdentifiable && !protectedIds.has(value.record.id),
      )
      .sort(describedOrder)
      .reverse();
    for (const value of removable.slice(0, needed)) selected.delete(value.record.id);
  }
}

function connectedComponents(
  records: DescribedRecord[],
  interventionIds: Map<string, string>,
): Map<string, DescribedRecord[]> {
  const parent = new Map(records.map((value) => [value.record.id, value.record.id]));
  const find = (id: string): string => {
    const current = required(parent, id, 'union parent');
    if (current === id) return id;
    const root = find(current);
    parent.set(id, root);
    return root;
  };
  const union = (left: string, right: string): void => {
    const leftRoot = find(left);
    const rightRoot = find(right);
    if (leftRoot === rightRoot) return;
    if (leftRoot < rightRoot) parent.set(rightRoot, leftRoot);
    else parent.set(leftRoot, rightRoot);
  };
  for (const values of groupBy(records, (value) => value.record.split_group).values()) {
    values.slice(1).forEach((value) => union(values[0].record.id, value.record.id));
  }
  const withIntervention = records.filter((value) => interventionIds.has(value.record.id));
  for (const values of groupBy(
    withIntervention,
    (value) => required(interventionIds, value.record.id, 'intervention'),
  ).values()) {
    values.slice(1).forEach((value) => union(values[0].record.id, value.record.id));
  }
  return groupBy(records, (value) => find(value.record.id));
}

function componentFeatures(
  values: DescribedRecord[],
  interventions: Map<string, string>,
): Record<string, number> {
  const features: Record<string, number> = { records: values.length };
  for (const value of values) {
    increment(features, `class:${value.record.target.identifiable ? 'identifiable' : 'ambiguous'}`);
    increment(features, `mechanic:${value.mechanic}`);
    increment(features, `evidence:${evidenceVariant(value.record.source_scenario_ids)}`);
  }
  const groups = new Set(values.flatMap((value) => interventions.get(value.record.id) ?? []));
  features.evidence_intervention_groups = groups.size;
  return features;
}

function toSurfaceRecords(
  value: DescribedRecord,
  split: V6Split,
  interventionId: string | null,
  surfaces: V5SurfaceVariant[],
): V6IdentifiabilityRecord[] {
  const canonicalInput = v6CanonicalInput(value.record.agent_input);
  const pairId = `v6-surface:${shortHash(canonicalInput, 24)}`;
  const context = `v6-context:${shortHash(value.record.split_group, 24)}`;
  return surfaces.map((surface): V6IdentifiabilityRecord => ({
    id: `identifiability-v6:${shortHash([split, pairId, surface], 24)}`,
    schema_version: 6,
    split,
    split_group: context,
    base_record_id: value.record.id,
    base_context_group: value.record.split_group,
    surface_pair_id: pairId,
    surface_variant: surface,
    invariance_group_id: pairId,
    evidence_intervention_id: interventionId,
    evidence_variant: evidenceVariant(value.record.source_scenario_ids),
    mechanic: value.mechanic,
    scenario_seeds: value.seeds,
    source_scenario_ids: [...value.record.source_scenario_ids],
    agent_input: transformV6Surface(canonicalInput, surface),
    target: {
      ambiguous: !value.record.target.identifiable,
      invariance: 'same_label_across_surfaces',
    },
  }));
}

export function v6CanonicalInput(input: AgentEpistemicInput): AgentEpistemicInput {
  const transformed = structuredClone(input);
  transformed.goal = 'bring the station beacon back online before the repeating hour begins again';
  return transformed;
}

export function transformV6Surface(
  canonicalInput: AgentEpistemicInput,
  surface: V5SurfaceVariant,
): AgentEpistemicInput {
  const transformed = transformSurface(canonicalInput, surface);
  if (surface === 'paraphrased') {
    transformed.goal = 'reactivate the station signal before the time loop starts over';
  }
  return transformed;
}

function describe(records: AgentEpistemicRecord[]): DescribedRecord[] {
  return records.map((record) => {
    const descriptor = sourceDescriptor(record.source_scenario_ids);
    return { record, ...descriptor };
  });
}

function sourceDescriptor(sourceScenarioIds: string[]): { mechanic: V6Mechanic; seeds: number[] } {
  const mechanics = new Set<V6Mechanic>();
  const seeds = new Set<number>();
  for (const id of sourceScenarioIds) {
    const match = id.match(/-(relockshort|powertrip|mirrorreject)-(\d+)-(?:trap|control)$/);
    if (!match) throw new Error(`Invalid V6 source scenario ${id}.`);
    mechanics.add(match[1] as V6Mechanic);
    seeds.add(Number(match[2]));
  }
  if (mechanics.size !== 1 || seeds.size === 0) {
    throw new Error(`V6 record crosses mechanics or has no seed: ${sourceScenarioIds.join(', ')}`);
  }
  return { mechanic: [...mechanics][0], seeds: [...seeds].sort((left, right) => left - right) };
}

function countBaseBySplit(records: V6IdentifiabilityRecord[]): Record<V6Split, number> {
  return mapSplits((split) => new Set(records.filter((record) => record.split === split).map((record) => record.surface_pair_id)).size);
}

function countInterventionsBySplit(records: V6IdentifiabilityRecord[]): Record<V6Split, number> {
  return mapSplits((split) => new Set(records.filter((record) => record.split === split).flatMap((record) => record.evidence_intervention_id ?? [])).size);
}

function countInterventionsByMechanic(
  records: V6IdentifiabilityRecord[],
): Record<V6Split, Partial<Record<V6Mechanic, number>>> {
  return mapSplits((split) => Object.fromEntries(
    (['relockshort', 'powertrip', 'mirrorreject'] as V6Mechanic[]).map((mechanic) => [
      mechanic,
      new Set(records.filter((record) => record.split === split && record.mechanic === mechanic).flatMap((record) => record.evidence_intervention_id ?? [])).size,
    ]),
  ));
}

function mapSplits<T>(value: (split: V6Split) => T): Record<V6Split, T> {
  return Object.fromEntries(
    (['train', 'calibration', 'mechanic_holdout'] as V6Split[]).map((split) => [split, value(split)]),
  ) as Record<V6Split, T>;
}

function increment(values: Record<string, number>, key: string): void {
  values[key] = (values[key] ?? 0) + 1;
}

function required<K, V>(values: Map<K, V>, key: K, label: string): V {
  const value = values.get(key);
  if (value === undefined) throw new Error(`Missing ${label} for ${String(key)}.`);
  return value;
}

function groupBy<T>(values: T[], key: (value: T) => string): Map<string, T[]> {
  const groups = new Map<string, T[]>();
  for (const value of values) groups.set(key(value), [...(groups.get(key(value)) ?? []), value]);
  return groups;
}

function baseRecordOrder(left: AgentEpistemicRecord, right: AgentEpistemicRecord): number {
  return canonicalJson([left.split_group, left.id]).localeCompare(canonicalJson([right.split_group, right.id]));
}

function describedOrder(left: DescribedRecord, right: DescribedRecord): number {
  return baseRecordOrder(left.record, right.record);
}

function recordOrder(left: V6IdentifiabilityRecord, right: V6IdentifiabilityRecord): number {
  return canonicalJson([left.split, left.split_group, left.id]).localeCompare(
    canonicalJson([right.split, right.split_group, right.id]),
  );
}
