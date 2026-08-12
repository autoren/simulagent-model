import type {
  AgentEpistemicInput,
  AgentEpistemicRecord,
  V5EvidenceVariant,
  V5SurfaceVariant,
  V7DevelopmentMechanic,
  V7IdentifiabilityRecord,
  V7Mechanic,
  V7Split,
} from './contracts';
import { canonicalJson, shortHash } from './serialization';
import { createStratifiedSplitPlan, type StratifiedGroup } from './stratified-split';
import { transformV6Surface, v6CanonicalInput } from './v6';

export interface V7SourceCandidate {
  record: AgentEpistemicRecord;
  mechanic: V7Mechanic;
  evidence: Exclude<V5EvidenceVariant, 'mixed'> | 'mixed';
  seeds: number[];
}

interface CurriculumCandidate extends V7SourceCandidate {
  internal_id: string;
  base_id: string;
  context_id: string;
  action_template: string;
  input: AgentEpistemicInput;
  ambiguous: boolean;
  intervention_id: string;
  intervention_kind: 'causal_rule_invariance' | 'oracle_label_change';
  selected_split: V7Split | null;
}

export interface V7BuildResult {
  records: V7IdentifiabilityRecord[];
  base_records: Record<V7Split, number>;
  candidate_records: {
    development: number;
    untouched_mechanic: number;
  };
  selected_oracle_label_change_groups: Record<V7Split, number>;
  stratification: {
    objective: number;
    restarts: number;
    calibration_ratio: number;
    connected_components: number;
  };
}

const developmentSplits: Array<'train' | 'calibration'> = ['train', 'calibration'];
const allSplits: V7Split[] = ['train', 'calibration', 'untouched_mechanic'];

export function buildV7Records(options: {
  developmentCandidates: V7SourceCandidate[];
  holdoutCandidates: V7SourceCandidate[];
  evidenceVariants: Exclude<V5EvidenceVariant, 'mixed'>[];
  surfaceVariants: V5SurfaceVariant[];
  calibrationRatio: number;
  stratificationRestarts: number;
  maximumPairsPerConditionalStratum: number;
  minimumLabelChangingDevelopmentGroups: number;
}): V7BuildResult {
  if (options.developmentCandidates.some((candidate) => candidate.mechanic === 'tonedrift')) {
    throw new Error('Tone drift must remain untouched during V7 development.');
  }
  if (options.holdoutCandidates.some((candidate) => candidate.mechanic !== 'tonedrift')) {
    throw new Error('The V7 untouched partition may contain only tone drift.');
  }

  const curriculum = makeDevelopmentCurriculum(
    options.developmentCandidates,
    options.evidenceVariants,
  );
  const components = connectedComponents(curriculum);
  const stratified = createStratifiedSplitPlan(
    [...components.entries()].map(([id, values]): StratifiedGroup => ({
      id,
      features: componentFeatures(values),
    })),
    { train: 1 - options.calibrationRatio, valid: options.calibrationRatio, test: 0 },
    'v7-causal-evidence-connected-split',
    options.stratificationRestarts,
  );
  const componentForCandidate = new Map<string, string>();
  for (const [component, values] of components) {
    for (const value of values) componentForCandidate.set(value.internal_id, component);
  }
  const partitioned = new Map<V7Split, CurriculumCandidate[]>([
    ['train', []],
    ['calibration', []],
  ]);
  for (const value of curriculum) {
    const component = required(componentForCandidate, value.internal_id, 'component');
    const split = required(stratified.plan, component, 'split') === 'valid' ? 'calibration' : 'train';
    partitioned.get(split)?.push(value);
  }

  const selectedDevelopment = developmentSplits.flatMap((split) => selectBalancedPartition(
    partitioned.get(split) ?? [],
    split,
    options.maximumPairsPerConditionalStratum,
    split === 'train'
      ? options.minimumLabelChangingDevelopmentGroups
      : Math.max(1, Math.floor(options.minimumLabelChangingDevelopmentGroups / 2)),
  ));
  const selectedHoldout = selectHoldout(options.holdoutCandidates);
  const base = [...selectedDevelopment, ...selectedHoldout];
  const records = base.flatMap((value) => toSurfaceRecords(value, options.surfaceVariants)).sort(recordOrder);

  return {
    records,
    base_records: mapSplits((split) => base.filter((value) => splitFor(value) === split).length),
    candidate_records: {
      development: curriculum.length,
      untouched_mechanic: options.holdoutCandidates.length,
    },
    selected_oracle_label_change_groups: mapSplits((split) => new Set(
      base
        .filter((value) => splitFor(value) === split && value.intervention_kind === 'oracle_label_change')
        .map((value) => value.intervention_id),
    ).size),
    stratification: {
      objective: stratified.objective,
      restarts: stratified.restarts,
      calibration_ratio: options.calibrationRatio,
      connected_components: components.size,
    },
  };
}

function makeDevelopmentCurriculum(
  candidates: V7SourceCandidate[],
  evidenceVariants: Exclude<V5EvidenceVariant, 'mixed'>[],
): CurriculumCandidate[] {
  const result: CurriculumCandidate[] = [];
  const forced = deduplicateInputs(candidates.filter((candidate) =>
    candidate.evidence === 'forced' && isCurriculumAction(candidate),
  ));
  for (const source of forced) {
    const baseId = `v7-card-base:${shortHash([source.mechanic, source.record.id], 24)}`;
    const contextId = `v7-context:${shortHash(source.record.split_group, 24)}`;
    const interventionId = `v7-rule:${shortHash([source.mechanic, source.record.id], 24)}`;
    for (const evidence of evidenceVariants) {
      const input = appendRelationalEvidenceLedger(
        v6CanonicalInput(source.record.agent_input),
        source.mechanic as V7DevelopmentMechanic,
        evidence,
        !source.record.target.identifiable,
      );
      result.push({
        ...source,
        evidence,
        internal_id: `v7-card:${shortHash([baseId, evidence], 24)}`,
        base_id: baseId,
        context_id: contextId,
        action_template: actionTemplate(source.record.agent_input.candidate_action.key),
        input,
        ambiguous: !source.record.target.identifiable,
        intervention_id: interventionId,
        intervention_kind: 'causal_rule_invariance',
        selected_split: null,
      });
    }
  }

  const bySignature = groupBy(candidates, evidenceSignature);
  for (const [signature, values] of bySignature) {
    const explicit = deduplicateInputs(values.filter((value) =>
      value.evidence !== 'mixed' && isCurriculumAction(value),
    ));
    const ambiguous = explicit.filter((value) => !value.record.target.identifiable).sort(sourceOrder);
    const identifiable = explicit.filter((value) => value.record.target.identifiable).sort(sourceOrder);
    const pair = firstCrossEvidencePair(ambiguous, identifiable);
    if (!pair) continue;
    const interventionId = `v7-oracle:${shortHash(signature, 24)}`;
    for (const source of pair) {
      const input = appendInvariantMemory(
        appendRelationalEvidenceLedger(
          v6CanonicalInput(source.record.agent_input),
          source.mechanic as V7DevelopmentMechanic,
          source.evidence as Exclude<V5EvidenceVariant, 'mixed'>,
          !source.record.target.identifiable,
        ),
        'Oracle contrast grouping: paired source evidence.',
      );
      result.push({
        ...source,
        internal_id: `v7-oracle-record:${shortHash([interventionId, source.record.id], 24)}`,
        base_id: `v7-oracle-base:${shortHash([interventionId, source.record.id], 24)}`,
        context_id: `v7-context:${shortHash(source.record.split_group, 24)}`,
        action_template: actionTemplate(source.record.agent_input.candidate_action.key),
        input,
        ambiguous: !source.record.target.identifiable,
        intervention_id: interventionId,
        intervention_kind: 'oracle_label_change',
        selected_split: null,
      });
    }
  }
  return deduplicateCurriculum(result);
}

function selectBalancedPartition(
  candidates: CurriculumCandidate[],
  split: 'train' | 'calibration',
  maximumPairs: number,
  minimumOracleGroups: number,
): CurriculumCandidate[] {
  const availableByCell = groupBy(candidates, conditionalCell);
  const balanceable = new Set(
    [...availableByCell.entries()]
      .filter(([, values]) => labels(values).size === 2)
      .map(([cell]) => cell),
  );
  const oracleGroups = [...groupBy(
    candidates.filter((value) => value.intervention_kind === 'oracle_label_change'),
    (value) => value.intervention_id,
  ).values()]
    .filter((values) => labels(values).size === 2 && values.every((value) => balanceable.has(conditionalCell(value))))
    .sort((left, right) => left[0].intervention_id.localeCompare(right[0].intervention_id));
  if (oracleGroups.length < minimumOracleGroups) {
    throw new Error(`V7 ${split} has only ${oracleGroups.length} balanceable oracle label-change groups.`);
  }
  const protectedIds = new Set(
    oracleGroups.slice(0, minimumOracleGroups).flatMap((values) => values.map((value) => value.internal_id)),
  );
  const selected: CurriculumCandidate[] = [];
  for (const [cell, values] of [...availableByCell.entries()].sort(([left], [right]) => left.localeCompare(right))) {
    if (!balanceable.has(cell)) continue;
    const byLabel = new Map<boolean, CurriculumCandidate[]>([
      [false, values.filter((value) => !value.ambiguous).sort(curriculumOrder)],
      [true, values.filter((value) => value.ambiguous).sort(curriculumOrder)],
    ]);
    const protectedByLabel = new Map<boolean, CurriculumCandidate[]>([
      [false, (byLabel.get(false) ?? []).filter((value) => protectedIds.has(value.internal_id))],
      [true, (byLabel.get(true) ?? []).filter((value) => protectedIds.has(value.internal_id))],
    ]);
    const target = Math.max(
      protectedByLabel.get(false)?.length ?? 0,
      protectedByLabel.get(true)?.length ?? 0,
      Math.min(
        maximumPairs,
        byLabel.get(false)?.length ?? 0,
        byLabel.get(true)?.length ?? 0,
      ),
    );
    for (const label of [false, true]) {
      const protectedValues = protectedByLabel.get(label) ?? [];
      const fillers = (byLabel.get(label) ?? []).filter((value) => !protectedIds.has(value.internal_id));
      if (protectedValues.length > target || protectedValues.length + fillers.length < target) {
        throw new Error(`V7 cannot balance ${split} conditional cell ${cell}.`);
      }
      selected.push(...protectedValues, ...fillers.slice(0, target - protectedValues.length));
    }
  }
  return selected.map((value) => ({ ...value, selected_split: split }));
}

function selectHoldout(sources: V7SourceCandidate[]): CurriculumCandidate[] {
  const values = deduplicateInputs(sources).map((source): CurriculumCandidate => {
    const signature = evidenceSignature(source);
    const interventionId = `v7-holdout-evidence:${shortHash(signature, 24)}`;
    return {
      ...source,
      internal_id: `v7-holdout:${shortHash(source.record.id, 24)}`,
      base_id: `v7-holdout-base:${shortHash(source.record.id, 24)}`,
      context_id: `v7-holdout-context:${shortHash(source.record.split_group, 24)}`,
      action_template: actionTemplate(source.record.agent_input.candidate_action.key),
      input: appendInvariantMemory(
        v6CanonicalInput(source.record.agent_input),
        'Resonance audit: mirror seating, generator power, and carried-tone calibration are separate causal states.',
      ),
      ambiguous: !source.record.target.identifiable,
      intervention_id: interventionId,
      intervention_kind: 'causal_rule_invariance',
      selected_split: 'untouched_mechanic',
    };
  });
  const groups = groupBy(values, (value) => value.intervention_id);
  const protectedValues: CurriculumCandidate[] = [];
  for (const group of [...groups.values()].sort((left, right) => left[0].intervention_id.localeCompare(right[0].intervention_id))) {
    const pair = firstCurriculumCrossEvidencePair(
      group.filter((value) => value.ambiguous).sort(curriculumOrder),
      group.filter((value) => !value.ambiguous).sort(curriculumOrder),
    );
    if (!pair) continue;
    for (const value of pair) value.intervention_kind = 'oracle_label_change';
    protectedValues.push(...pair);
  }
  if (new Set(protectedValues.map((value) => value.intervention_id)).size === 0) {
    throw new Error('The untouched tone-drift mechanic has no oracle label-changing evidence pair.');
  }
  const protectedIds = new Set(protectedValues.map((value) => value.internal_id));
  const ambiguous = values.filter((value) => value.ambiguous && !protectedIds.has(value.internal_id)).sort(curriculumOrder);
  const identifiable = values.filter((value) => !value.ambiguous && !protectedIds.has(value.internal_id)).sort(holdoutIdentifiableOrder);
  const protectedCounts = {
    ambiguous: protectedValues.filter((value) => value.ambiguous).length,
    identifiable: protectedValues.filter((value) => !value.ambiguous).length,
  };
  const targetPerLabel = Math.min(
    48,
    protectedCounts.ambiguous + ambiguous.length,
    protectedCounts.identifiable + identifiable.length,
  );
  return [
    ...protectedValues,
    ...ambiguous.slice(0, targetPerLabel - protectedCounts.ambiguous),
    ...identifiable.slice(0, targetPerLabel - protectedCounts.identifiable),
  ].sort(curriculumOrder);
}

function toSurfaceRecords(
  value: CurriculumCandidate,
  surfaces: V5SurfaceVariant[],
): V7IdentifiabilityRecord[] {
  const split = splitFor(value);
  const pairId = `v7-surface:${shortHash([split, value.base_id, canonicalJson(value.input)], 24)}`;
  return surfaces.map((surface): V7IdentifiabilityRecord => ({
    id: `identifiability-v7:${shortHash([split, pairId, surface], 24)}`,
    schema_version: 7,
    split,
    split_group: value.context_id,
    base_record_id: value.base_id,
    base_context_group: value.record.split_group,
    surface_pair_id: pairId,
    surface_variant: surface,
    invariance_group_id: pairId,
    evidence_intervention_id: value.intervention_id,
    evidence_intervention_kind: value.intervention_kind,
    evidence_variant: value.evidence,
    mechanic: value.mechanic,
    action_template: value.action_template,
    scenario_seeds: [...value.seeds],
    source_scenario_ids: [...value.record.source_scenario_ids],
    agent_input: transformV6Surface(value.input, surface),
    target: {
      ambiguous: value.ambiguous,
      invariance: 'same_label_across_surfaces',
    },
  }));
}

function appendRelationalEvidenceLedger(
  input: AgentEpistemicInput,
  mechanic: V7DevelopmentMechanic,
  evidence: Exclude<V5EvidenceVariant, 'mixed'>,
  ambiguous: boolean,
): AgentEpistemicInput {
  const frames: Record<Exclude<V5EvidenceVariant, 'mixed'>, string> = {
    forced: 'Causal audit:',
    announced: 'Direct-evidence rule:',
    'announced-upstream': 'Upstream-evidence rule:',
    'announced-consequence': 'Consequence rule:',
    'announced-procedure': 'Procedure rule:',
    unobservable: 'Observability rule:',
  };
  // The two ledgers have the exact same bag of words: two "confirmed" and two
  // "unresolved" tokens. Only their assignment to causal roles differs. The assignment
  // has opposite labels across mechanics, so neither a unigram shortcut nor a fixed
  // ledger pattern can solve the task without relating action, mechanic, and evidence.
  const patternA = {
    hatch: 'confirmed',
    generator: 'unresolved',
    mirror: 'unresolved',
    weather: 'confirmed',
  } as const;
  const patternB = {
    hatch: 'unresolved',
    generator: 'confirmed',
    mirror: 'confirmed',
    weather: 'unresolved',
  } as const;
  const usePatternA = mechanic === 'relockshort' ? !ambiguous : ambiguous;
  const pattern = usePatternA ? patternA : patternB;
  const ledger = `${frames[evidence]} status ledger; ` +
    `hatch state ${pattern.hatch}; generator state ${pattern.generator}; ` +
    `mirror state ${pattern.mirror}; weather state ${pattern.weather}`;
  return appendInvariantMemory(input, ledger);
}

function isCurriculumAction(value: V7SourceCandidate): boolean {
  const template = actionTemplate(value.record.agent_input.candidate_action.key);
  const required: Record<V7DevelopmentMechanic, Set<string>> = {
    relockshort: new Set(['move:up', 'use:key', 'inspect:hatch']),
    powertrip: new Set(['use:tone', 'use:mirror', 'inspect:power']),
  };
  return required[value.mechanic as V7DevelopmentMechanic]?.has(template) ?? false;
}

function appendInvariantMemory(input: AgentEpistemicInput, memory: string): AgentEpistemicInput {
  const result = structuredClone(input);
  result.observation.memories = [...result.observation.memories, memory].slice(-16);
  return result;
}

export function actionTemplate(key: string): string {
  const [verb, target = ''] = key.split(':');
  if (verb === 'use') {
    const roles: Record<string, string> = {
      brassKey: 'key',
      tuningFork: 'tone',
      mirrorShard: 'mirror',
    };
    return `use:${roles[target] ?? target}`;
  }
  if (verb === 'move') return `move:${target}`;
  if (verb === 'inspect') {
    if (target === 'room') return 'inspect:room';
    if (target === 'sealedHatch') return 'inspect:hatch';
    if (target === 'coilBank' || target === 'maintenancePanel') return 'inspect:power';
    if (target === 'beaconConsole' || target === 'starLens') return 'inspect:beacon';
    return 'inspect:other';
  }
  if (verb === 'take') return 'take:resource';
  if (verb === 'talk') return 'talk:character';
  return verb;
}

function evidenceSignature(value: V7SourceCandidate): string {
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

function conditionalCell(value: CurriculumCandidate): string {
  return canonicalJson([value.mechanic, value.evidence, value.action_template]);
}

function connectedComponents(values: CurriculumCandidate[]): Map<string, CurriculumCandidate[]> {
  const parent = new Map(values.map((value) => [value.internal_id, value.internal_id]));
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
  for (const group of [
    ...groupBy(values, (value) => value.context_id).values(),
    ...groupBy(values, (value) => value.intervention_id).values(),
  ]) {
    group.slice(1).forEach((value) => union(group[0].internal_id, value.internal_id));
  }
  return groupBy(values, (value) => find(value.internal_id));
}

function componentFeatures(values: CurriculumCandidate[]): Record<string, number> {
  const features: Record<string, number> = { records: values.length };
  for (const value of values) {
    increment(features, `cell:${conditionalCell(value)}:label:${value.ambiguous ? 'ambiguous' : 'identifiable'}`);
    increment(features, `mechanic:${value.mechanic}`);
    if (value.intervention_kind === 'oracle_label_change') increment(features, 'oracle_label_change_records');
  }
  return features;
}

function firstCrossEvidencePair(
  ambiguous: V7SourceCandidate[],
  identifiable: V7SourceCandidate[],
): [V7SourceCandidate, V7SourceCandidate] | null {
  for (const left of ambiguous) {
    const right = identifiable.find((value) => value.evidence !== left.evidence);
    if (right) return [left, right];
  }
  return null;
}

function firstCurriculumCrossEvidencePair(
  ambiguous: CurriculumCandidate[],
  identifiable: CurriculumCandidate[],
): [CurriculumCandidate, CurriculumCandidate] | null {
  for (const left of ambiguous) {
    const right = identifiable.find((value) => value.evidence !== left.evidence);
    if (right) return [left, right];
  }
  return null;
}

function deduplicateInputs(values: V7SourceCandidate[]): V7SourceCandidate[] {
  const byInput = new Map<string, V7SourceCandidate>();
  for (const value of [...values].sort(sourceOrder)) {
    const key = canonicalJson(value.record.agent_input);
    const prior = byInput.get(key);
    if (!prior) {
      byInput.set(key, value);
    } else if (prior.record.target.identifiable !== value.record.target.identifiable) {
      throw new Error('Oracle candidates assign different labels to an identical visible input.');
    }
  }
  return [...byInput.values()];
}

function deduplicateCurriculum(values: CurriculumCandidate[]): CurriculumCandidate[] {
  const byInput = new Map<string, CurriculumCandidate>();
  for (const value of [...values].sort(curriculumOrder)) {
    const key = canonicalJson(value.input);
    const prior = byInput.get(key);
    if (!prior) byInput.set(key, value);
    else if (prior.ambiguous !== value.ambiguous) {
      throw new Error('V7 curriculum assigns different labels to an identical model input.');
    }
  }
  return [...byInput.values()];
}

function labels(values: CurriculumCandidate[]): Set<boolean> {
  return new Set(values.map((value) => value.ambiguous));
}

function splitFor(value: CurriculumCandidate): V7Split {
  if (value.selected_split === null) {
    throw new Error(`Candidate ${value.internal_id} has no selected split.`);
  }
  return value.selected_split;
}

function sourceOrder(left: V7SourceCandidate, right: V7SourceCandidate): number {
  return canonicalJson([left.mechanic, left.evidence, left.record.split_group, left.record.id]).localeCompare(
    canonicalJson([right.mechanic, right.evidence, right.record.split_group, right.record.id]),
  );
}

function curriculumOrder(left: CurriculumCandidate, right: CurriculumCandidate): number {
  return canonicalJson([
    left.mechanic,
    left.evidence,
    left.action_template,
    left.intervention_kind,
    left.internal_id,
  ]).localeCompare(canonicalJson([
    right.mechanic,
    right.evidence,
    right.action_template,
    right.intervention_kind,
    right.internal_id,
  ]));
}

function holdoutIdentifiableOrder(left: CurriculumCandidate, right: CurriculumCandidate): number {
  const evidence = left.evidence.localeCompare(right.evidence);
  return evidence || curriculumOrder(left, right);
}

function recordOrder(left: V7IdentifiabilityRecord, right: V7IdentifiabilityRecord): number {
  return canonicalJson([left.split, left.split_group, left.id]).localeCompare(
    canonicalJson([right.split, right.split_group, right.id]),
  );
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

function mapSplits<T>(value: (split: V7Split) => T): Record<V7Split, T> {
  return Object.fromEntries(allSplits.map((split) => [split, value(split)])) as Record<V7Split, T>;
}
