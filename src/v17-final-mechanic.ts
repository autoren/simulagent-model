import type { Action, WorldState } from '../../simulagent/src/simulation';
import { createInitialState, resolveAction } from '../../simulagent/src/simulation';
import type { V8StructuredRecord, V8SurfaceVariant } from './contracts';
import type { V14GroundingRecord } from './v14-contracts';
import { buildV14GroundingRecords, v14SurfaceFamilies } from './v14-grounding';
import { evaluateAllowedTransitions } from './v9-symbolic';
import { canonicalJson, sha256, shortHash } from './serialization';

export const v17FinalMechanic = 'beacon_console_diagnostic';
export const v17LexicalReferenceMechanic = 'beacon_calibration';
export const v17OperatorFamily = 'multiway_partition';
export const v17Determinants = ['generator_stable', 'mirror_seated', 'fork_calibrated'] as const;

type Assignment = Record<(typeof v17Determinants)[number], boolean>;
type V17FinalRecord = Omit<V14GroundingRecord, 'schema_version'> & { schema_version: 17 };

export interface V17ExpectedTopology {
  sourceScaffolds: number;
  records: number;
  contexts: number;
  interventionGroups: number;
  recordsPerTemplateLexiconCell: number;
  transitionCases: number;
  transitionCodes: number;
}

export interface V17BuildResult {
  records: V17FinalRecord[];
  source_scaffolds: number;
  transition_table_sha256: string;
  validation: V17Validation;
}

export interface V17Validation {
  errors: string[];
  records: number;
  contexts: number;
  intervention_groups: number;
  ambiguous_records: number;
  identifiable_records: number;
  templates: Record<string, number>;
  lexicons: Record<string, number>;
  semantic_operators: Record<string, number>;
  template_lexicon_cells: Record<string, number>;
  transition_cases: number;
  transition_codes: number;
  symbolic_mismatches: number;
  malformed_intervention_groups: number;
  duplicate_ids: number;
  exact_span_mismatches: number;
  answer_bearing_inputs: number;
  development_candidate_action_overlaps: number;
  forbidden_source_records: number;
}

const action: Action = { type: 'inspect', target: 'beaconConsole' };
const actionLabels: Record<V8SurfaceVariant, string> = {
  canonical: 'inspect the beacon console diagnostic readout',
  entity_renamed: 'query station gamma diagnostic panel',
  paraphrased: 'read the observatory console status display',
};

export function buildV17FinalMechanic(
  sourceRecords: V8StructuredRecord[],
  sourceDatasetSha256: string,
  expected: V17ExpectedTopology,
  developmentCandidateActions: Set<string>,
  sourceReplica = 0,
): V17BuildResult {
  const source = sourceRecords.filter((record) =>
    record.mechanic === v17LexicalReferenceMechanic && record.replica === sourceReplica,
  );
  const table = transitionTable();
  const scaffolds = source.map((record) => replaceMechanic(record, table));
  const rendered = buildV14GroundingRecords(
    scaffolds,
    sourceDatasetSha256,
    sourceReplica,
  ).map((record): V17FinalRecord => ({
    ...record,
    id: `v17:${shortHash([v17FinalMechanic, record.id], 24)}`,
    schema_version: 17,
    split: 'evaluation',
    context_group: `v17-context:${shortHash([v17FinalMechanic, record.context_group], 24)}`,
    complement_group: `v17-complement:${shortHash([v17FinalMechanic, record.complement_group], 24)}`,
    intervention_group_id: `v17-intervention:${shortHash([v17FinalMechanic, record.intervention_group_id], 24)}`,
    mechanic: v17FinalMechanic,
    operator_family: v17OperatorFamily,
  })).sort(recordOrder);
  const validation = validateV17FinalMechanic(rendered, expected, developmentCandidateActions);
  if (source.length !== expected.sourceScaffolds) {
    validation.errors.push(`Expected ${expected.sourceScaffolds} source scaffolds, found ${source.length}.`);
  }
  return {
    records: rendered,
    source_scaffolds: source.length,
    transition_table_sha256: sha256(canonicalJson(table)),
    validation,
  };
}

function transitionTable(): V8StructuredRecord['agent_input']['action_dependency_schema']['transition_cases'] {
  const values = booleanAssignments();
  const hashes = values.map(transitionHash);
  const codes = new Map(uniqueSorted(hashes).map((hash, index) => [
    hash,
    `transition_${String(index + 1).padStart(2, '0')}`,
  ]));
  if (codes.size !== 8) throw new Error(`V17 final mechanic expected 8 simulator outcomes, found ${codes.size}.`);
  return values.map((assignment, index) => ({
    values: v17Determinants.map((id) => assignment[id] ? 'active' as const : 'inactive' as const),
    transition_code: required(codes, hashes[index]),
  }));
}

function transitionHash(assignment: Assignment): string {
  const state = createInitialState('baseline');
  prepareDiagnosticState(state, assignment);
  const next = resolveAction(state, action, { driver: 'manual' });
  const latest = next.log.at(-1);
  if (!latest?.valid || !latest.actualOutcome || !latest.actionSurfaceDelta) {
    throw new Error('V17 final mechanic simulator transition is missing or invalid.');
  }
  return sha256(canonicalJson({
    visible_action_outcome: latest.outcome,
    actual_outcome: latest.actualOutcome,
    action_surface_delta: latest.actionSurfaceDelta,
  }));
}

function prepareDiagnosticState(state: WorldState, assignment: Assignment): void {
  state.location = 'observatory';
  state.rooms.observatory.visited = true;
  if (!state.inventory.includes('tuningFork')) state.inventory.push('tuningFork');
  state.flags.generatorStable = assignment.generator_stable;
  state.flags.mirrorShardInstalled = assignment.mirror_seated;
  state.flags.tuningForkDetuned = !assignment.fork_calibrated;
  state.flags.inspectedBeacon = false;
  state.agent.beliefs = [];
  state.agent.memories = [];
  state.stats.pressure = 0;
  state.stats.signal = assignment.mirror_seated ? 45 : 0;
}

function replaceMechanic(
  source: V8StructuredRecord,
  table: V8StructuredRecord['agent_input']['action_dependency_schema']['transition_cases'],
): V8StructuredRecord {
  const record = structuredClone(source);
  const assignment = record.oracle.actual_assignment as Assignment;
  const primary = record.primary_determinant_id as (typeof v17Determinants)[number];
  if (!v17Determinants.includes(primary)) throw new Error(`V17 scaffold has unexpected determinant ${primary}.`);
  const compatible = record.intervention_member === 'relevant_unresolved'
    ? booleanAssignments().filter((candidate) => v17Determinants.every((id) => id === primary || candidate[id] === assignment[id]))
    : [assignment];
  const tableLookup = new Map(table.map((value) => [value.values.join('|'), value.transition_code]));
  const possible = uniqueSorted(compatible.map((candidate) => required(
    tableLookup,
    v17Determinants.map((id) => candidate[id] ? 'active' : 'inactive').join('|'),
  )));
  const ambiguous = possible.length > 1;
  const labels = actionLabels[record.surface_variant];
  const groupKey = [v17FinalMechanic, canonicalJson(assignment), primary, source.replica];
  const groupId = `v17-source-intervention:${shortHash(groupKey, 24)}`;
  const baseId = `v17-source-base:${shortHash([groupKey, record.intervention_member], 24)}`;
  record.id = `v17-source:${shortHash([baseId, record.surface_variant], 24)}`;
  record.split_group = `v17-source-context:${shortHash(groupKey, 24)}`;
  record.action_template = 'inspect:beacon-diagnostic';
  record.intervention_group_id = groupId;
  record.intervention_kind = 'oracle_label_flip';
  record.surface_group_id = `v17-source-surface:${shortHash(baseId, 24)}`;
  record.source_scenario_id = 'baseline';
  record.agent_input.candidate_action = { key: record.action_template, label: labels };
  record.agent_input.available_actions = [{ key: record.action_template, label: labels }];
  record.agent_input.action_dependency_schema = {
    ...record.agent_input.action_dependency_schema,
    candidate_action: labels,
    transition_cases: table,
  };
  record.target.ambiguous = ambiguous;
  record.target.possible_transition_count = possible.length;
  record.target.determinant_ledger = record.target.determinant_ledger.map((value, index) => {
    if (index >= v17Determinants.length) return { ...value, status: 'IRRELEVANT' };
    const id = v17Determinants[index];
    if (id === primary && record.intervention_member === 'relevant_unresolved') {
      return { ...value, status: 'UNRESOLVED_OUTCOME_SENSITIVE' };
    }
    return { ...value, status: assignment[id] ? 'RESOLVED_TRUE' : 'RESOLVED_FALSE' };
  });
  record.target.decisive_unresolved_determinants = record.intervention_member === 'relevant_unresolved'
    ? [primary]
    : [];
  record.oracle.compatible_assignments = compatible.length;
  record.oracle.possible_transition_sha256 = possible;
  return record;
}

export function validateV17FinalMechanic(
  records: V17FinalRecord[],
  expected: V17ExpectedTopology,
  developmentCandidateActions: Set<string>,
): V17Validation {
  const errors: string[] = [];
  const countBy = (key: (record: V17FinalRecord) => string) => Object.fromEntries(
    [...new Set(records.map(key))].sort().map((value) => [value, records.filter((record) => key(record) === value).length]),
  );
  const templates = countBy((record) => record.template_family);
  const lexicons = countBy((record) => record.state_lexicon_family);
  const semanticOperators = countBy((record) => record.semantic_operator_family);
  const cells = countBy((record) => `${record.template_family}:${record.state_lexicon_family}`);
  const groups = new Map<string, V17FinalRecord[]>();
  for (const record of records) {
    const values = groups.get(record.intervention_group_id) ?? [];
    values.push(record);
    groups.set(record.intervention_group_id, values);
  }
  let symbolicMismatches = 0;
  let exactSpanMismatches = 0;
  let answerBearingInputs = 0;
  let forbiddenSourceRecords = 0;
  for (const record of records) {
    const symbolic = evaluateAllowedTransitions({
      action_dependency_schema: record.action_dependency_schema,
      determinant_values: record.target.determinant_grounding,
    });
    if (symbolic.identifiable !== record.target.identifiable ||
        canonicalJson(symbolic.possible_transition_codes) !== canonicalJson(record.target.possible_transition_codes)) {
      symbolicMismatches += 1;
    }
    for (const target of record.target.determinant_grounding) {
      if (record.agent_input.observation.slice(target.evidence_span.start, target.evidence_span.end) !== target.evidence_span.text) {
        exactSpanMismatches += 1;
      }
    }
    if (/\b(?:active|inactive|entailed|contradicted|unknown)\b/i.test(record.agent_input.observation)) answerBearingInputs += 1;
    if (record.source.v8_record_id.includes('tonedrift') || canonicalJson(record).toLowerCase().includes('tone drift')) {
      forbiddenSourceRecords += 1;
    }
  }
  const malformedGroups = [...groups.values()].filter((values) =>
    values.length !== 6 ||
    new Set(values.map((value) => value.state_lexicon_family)).size !== 3 ||
    new Set(values.map((value) => value.intervention_member)).size !== 2 ||
    values.filter((value) => !value.target.identifiable).length !== 3 ||
    values.some((value) => value.intervention_kind !== 'oracle_label_flip'),
  ).length;
  const candidateActions = new Set(records.map((record) => record.agent_input.candidate_action));
  const actionOverlaps = [...candidateActions].filter((value) => developmentCandidateActions.has(value)).length;
  const transitionCases = records[0]?.action_dependency_schema.transition_cases.length ?? 0;
  const transitionCodes = new Set(records[0]?.action_dependency_schema.transition_cases.map((value) => value.transition_code) ?? []).size;
  const ambiguous = records.filter((record) => !record.target.identifiable).length;
  const identifiers = records.map((record) => record.id);

  if (records.length !== expected.records) errors.push(`Expected ${expected.records} records, found ${records.length}.`);
  if (new Set(records.map((record) => record.context_group)).size !== expected.contexts) errors.push('V17 context count differs from the lock.');
  if (groups.size !== expected.interventionGroups) errors.push('V17 intervention-group count differs from the lock.');
  if (Object.keys(templates).length !== v14SurfaceFamilies.length) errors.push('V17 does not contain all nine template families.');
  if (Object.keys(lexicons).length !== 3) errors.push('V17 does not contain all three state lexicons.');
  if (Object.values(cells).some((value) => value !== expected.recordsPerTemplateLexiconCell)) errors.push('V17 template-by-lexicon cell size differs from the lock.');
  if (ambiguous * 2 !== records.length) errors.push('V17 identifiability labels are not balanced.');
  if (transitionCases !== expected.transitionCases || transitionCodes !== expected.transitionCodes) errors.push('V17 transition table is not the locked injective eight-way table.');
  if (symbolicMismatches) errors.push(`V17 has ${symbolicMismatches} symbolic target mismatches.`);
  if (malformedGroups) errors.push(`V17 has ${malformedGroups} malformed intervention groups.`);
  if (new Set(identifiers).size !== identifiers.length) errors.push('V17 has duplicate record ids.');
  if (exactSpanMismatches) errors.push(`V17 has ${exactSpanMismatches} exact-span mismatches.`);
  if (answerBearingInputs) errors.push(`V17 has ${answerBearingInputs} answer-bearing observations.`);
  if (actionOverlaps) errors.push('V17 candidate action overlaps development.');
  if (forbiddenSourceRecords) errors.push('V17 contains a forbidden protected source reference.');
  if (records.some((record) => record.mechanic !== v17FinalMechanic || record.operator_family !== v17OperatorFamily || record.split !== 'evaluation')) {
    errors.push('V17 record identity, operator, or split differs from the lock.');
  }

  return {
    errors,
    records: records.length,
    contexts: new Set(records.map((record) => record.context_group)).size,
    intervention_groups: groups.size,
    ambiguous_records: ambiguous,
    identifiable_records: records.length - ambiguous,
    templates,
    lexicons,
    semantic_operators: semanticOperators,
    template_lexicon_cells: cells,
    transition_cases: transitionCases,
    transition_codes: transitionCodes,
    symbolic_mismatches: symbolicMismatches,
    malformed_intervention_groups: malformedGroups,
    duplicate_ids: identifiers.length - new Set(identifiers).size,
    exact_span_mismatches: exactSpanMismatches,
    answer_bearing_inputs: answerBearingInputs,
    development_candidate_action_overlaps: actionOverlaps,
    forbidden_source_records: forbiddenSourceRecords,
  };
}

function booleanAssignments(): Assignment[] {
  return Array.from({ length: 8 }, (_, mask) => Object.fromEntries(
    v17Determinants.map((id, index) => [id, Boolean(mask & (1 << index))]),
  ) as Assignment);
}

function uniqueSorted(values: string[]): string[] {
  return [...new Set(values)].sort();
}

function required<K, V>(values: Map<K, V>, key: K): V {
  const value = values.get(key);
  if (value === undefined) throw new Error(`Missing V17 map value ${String(key)}.`);
  return value;
}

function recordOrder(left: V17FinalRecord, right: V17FinalRecord): number {
  return canonicalJson([
    left.template_family,
    left.state_lexicon_family,
    left.context_group,
    left.intervention_member,
  ]).localeCompare(canonicalJson([
    right.template_family,
    right.state_lexicon_family,
    right.context_group,
    right.intervention_member,
  ]));
}
