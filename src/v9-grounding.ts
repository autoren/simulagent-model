import type { V8StructuredRecord, V8SurfaceVariant } from './contracts';
import type {
  V9BooleanValue,
  V9EvidenceUnit,
  V9GroundingRecord,
  V9OperatorFamily,
  V9TemplateFamily,
  V9TemporalStatus,
} from './v9-contracts';
import { canonicalJson, shortHash } from './serialization';
import { evaluateAllowedTransitions } from './v9-symbolic';

interface PhraseSet {
  subject: string;
  active: string;
  inactive: string;
}

type UncertaintyMode = 'unknown' | 'stale' | 'conflicting';

const TEMPLATE_BY_REPLICA: V9TemplateFamily[] = [
  'inspection_report',
  'operator_log',
  'questioned_claim',
  'technical_summary',
];

const OPERATOR_BY_MECHANIC: Record<string, V9OperatorFamily> = {
  hatch_traversal: 'binary_partition',
  beacon_calibration: 'binary_partition',
  pressure_hatch_relock: 'binary_partition',
  generator_tuning: 'multiway_partition',
  mirror_power_trip: 'multiway_partition',
  mirror_rejection: 'multiway_partition',
};

export function buildV9GroundingRecords(
  sourceRecords: V8StructuredRecord[],
  sourceDatasetSha256: string,
  calibrationModulo = 4,
): V9GroundingRecord[] {
  const selected = sourceRecords.filter((record) => record.replica >= 0 && record.replica < 4);
  const contextsByMechanic = new Map<string, string[]>();
  for (const record of selected) {
    const key = semanticContextKey(record);
    const values = contextsByMechanic.get(record.mechanic) ?? [];
    if (!values.includes(key)) contextsByMechanic.set(record.mechanic, [...values, key]);
  }
  const calibrationContexts = new Set<string>();
  for (const values of contextsByMechanic.values()) {
    values.sort();
    values.forEach((value, index) => {
      if (index % calibrationModulo === 0) calibrationContexts.add(value);
    });
  }

  return selected.map((record) => createV9Record(
    record,
    sourceDatasetSha256,
    calibrationContexts.has(semanticContextKey(record)) ? 'calibration' : 'train',
  )).sort((left, right) => canonicalJson([
    left.split,
    left.mechanic,
    left.context_group,
    left.template_family,
    left.intervention_member,
    left.surface_variant,
  ]).localeCompare(canonicalJson([
    right.split,
    right.mechanic,
    right.context_group,
    right.template_family,
    right.intervention_member,
    right.surface_variant,
  ])));
}

function createV9Record(
  source: V8StructuredRecord,
  sourceDatasetSha256: string,
  split: 'train' | 'calibration',
): V9GroundingRecord {
  const schema = source.agent_input.action_dependency_schema;
  const relevantCount = schema.transition_determinants.length;
  const canonicalIds = source.target.determinant_ledger.slice(0, relevantCount).map((value) => value.id);
  const canonicalByModelId = new Map(schema.transition_determinants.map((value, index) => [
    value.id,
    canonicalIds[index],
  ]));
  const template = TEMPLATE_BY_REPLICA[source.replica];
  if (!template) throw new Error(`V9 has no template for replica ${source.replica}.`);
  const uncertaintyMode = uncertaintyModeFor(source);
  const rendered = source.agent_input.evidence_ledger.map((fact, index) => {
    const canonicalId = canonicalByModelId.get(fact.id);
    const phrases = canonicalId
      ? determinantPhrases(canonicalId, source.surface_variant, fact.id, fact.role)
      : irrelevantPhrases(index, source.surface_variant, fact.role);
    const allowedValues: V9BooleanValue[] = fact.evidence_state === 'unresolved'
      ? ['inactive', 'active']
      : [fact.value as V9BooleanValue];
    const temporalStatus: V9TemporalStatus = fact.evidence_state === 'confirmed'
      ? 'CURRENT'
      : uncertaintyMode === 'stale'
        ? 'STALE_ONLY'
        : uncertaintyMode === 'conflicting'
          ? 'CONFLICTING_CURRENT'
          : 'UNKNOWN_CURRENT';
    return {
      orderKey: shortHash([semanticContextKey(source), template, source.surface_variant, fact.id], 16),
      determinantId: canonicalId ? fact.id : null,
      allowedValues,
      temporalStatus,
      text: renderEvidence(phrases, fact.value, template, uncertaintyMode, index),
    };
  }).sort((left, right) => left.orderKey.localeCompare(right.orderKey));
  const evidenceUnits: V9EvidenceUnit[] = [];
  let observation = `Audit scene ${shortHash(semanticContextKey(source), 10)} contains seven independently recorded statements.`;
  for (const value of rendered) {
    observation += '\n';
    const start = observation.length;
    observation += value.text;
    evidenceUnits.push({ start, end: observation.length, text: value.text });
  }
  const determinantGrounding = schema.transition_determinants.map((determinant) => {
    const index = rendered.findIndex((value) => value.determinantId === determinant.id);
    if (index < 0) throw new Error(`V9 rendering omitted determinant ${determinant.id}.`);
    return {
      determinant_id: determinant.id,
      allowed_values: rendered[index].allowedValues,
      temporal_status: rendered[index].temporalStatus,
      evidence_span: evidenceUnits[index],
    };
  });
  const symbolic = evaluateAllowedTransitions({
    action_dependency_schema: schema,
    determinant_values: determinantGrounding,
  });
  const contextKey = semanticContextKey(source);
  const contextGroup = `v9-context:${shortHash(contextKey, 24)}`;
  const interventionGroup = `v9-intervention:${shortHash([contextKey, template], 24)}`;
  return {
    id: `v9:${shortHash([source.id, split], 24)}`,
    schema_version: 9,
    split,
    context_group: contextGroup,
    intervention_group_id: interventionGroup,
    intervention_kind: source.intervention_kind,
    intervention_member: source.intervention_member,
    mechanic: source.mechanic,
    operator_family: requiredOperator(source.mechanic),
    template_family: template,
    surface_variant: source.surface_variant,
    action_dependency_schema: schema,
    agent_input: {
      task: 'ground_transition_evidence',
      candidate_action: schema.candidate_action,
      transition_determinants: schema.transition_determinants,
      observation,
      output_instruction: 'For every listed determinant, return its allowed active/inactive values, temporal status, and exact supporting evidence span.',
    },
    evidence_units: evidenceUnits,
    target: {
      determinant_grounding: determinantGrounding,
      identifiable: symbolic.identifiable,
      possible_transition_codes: symbolic.possible_transition_codes,
    },
    source: {
      v8_record_id: source.id,
      v8_dataset_sha256: sourceDatasetSha256,
    },
  };
}

function semanticContextKey(record: V8StructuredRecord): string {
  return canonicalJson([
    record.mechanic,
    record.primary_determinant_id,
    record.oracle.actual_assignment,
  ]);
}

function uncertaintyModeFor(record: V8StructuredRecord): UncertaintyMode {
  const value = Number.parseInt(shortHash(semanticContextKey(record), 8), 16) % 3;
  return (['unknown', 'stale', 'conflicting'] as const)[value];
}

function renderEvidence(
  phrases: PhraseSet,
  value: 'active' | 'inactive' | 'hidden',
  template: V9TemplateFamily,
  mode: UncertaintyMode,
  row: number,
): string {
  if (value === 'hidden') return renderUnresolved(phrases, template, mode, row);
  const actual = value === 'active' ? phrases.active : phrases.inactive;
  const opposite = value === 'active' ? phrases.inactive : phrases.active;
  switch (template) {
    case 'inspection_report':
      return `A current inspection reports that ${actual}.`;
    case 'operator_log':
      return `The latest operator entry notes that ${actual}.`;
    case 'questioned_claim':
      return `A signed current check rejects the claim that ${opposite}.`;
    case 'technical_summary':
      return `Present-state summary: ${actual}.`;
  }
}

function renderUnresolved(
  phrases: PhraseSet,
  template: V9TemplateFamily,
  mode: UncertaintyMode,
  row: number,
): string {
  if (mode === 'stale') {
    const earlier = row % 2 === 0 ? phrases.active : phrases.inactive;
    const prefix = template === 'operator_log' ? 'An archived operator entry' : 'An earlier record';
    return `${prefix} said that ${earlier}, but no current reading is available.`;
  }
  if (mode === 'conflicting') {
    const prefix = template === 'technical_summary' ? 'The current technical reports conflict' : 'Two equally current reports disagree';
    return `${prefix}: one says that ${phrases.active}; the other says that ${phrases.inactive}.`;
  }
  if (template === 'questioned_claim') {
    return `No current evidence either supports or rejects the alternatives that ${phrases.active} or that ${phrases.inactive}.`;
  }
  return `The current record cannot determine whether ${phrases.active} or ${phrases.inactive}.`;
}

function determinantPhrases(
  id: string,
  surface: V8SurfaceVariant,
  modelId: string,
  role: string,
): PhraseSet {
  if (surface === 'entity_renamed') {
    const subject = role.replace(/^the /, '');
    return { subject, active: `${subject} is enabled`, inactive: `${subject} is disabled` };
  }
  const variants: Record<string, Record<'canonical' | 'paraphrased', PhraseSet>> = {
    hatch_unlocked: {
      canonical: { subject: 'observatory hatch', active: 'the observatory hatch stands unlatched', inactive: 'the observatory hatch remains latched' },
      paraphrased: { subject: 'upper passage', active: 'the upper passage can swing freely', inactive: 'the upper passage is held shut by its lock' },
    },
    generator_stable: {
      canonical: { subject: 'generator', active: 'the generator rhythm is even', inactive: 'the generator output surges unevenly' },
      paraphrased: { subject: 'coil bank', active: 'the coil bank carries steady power', inactive: 'the coil bank flickers between power levels' },
    },
    fork_calibrated: {
      canonical: { subject: 'tuning fork', active: 'the fork tone matches the reference pitch', inactive: 'the fork tone falls away from the reference pitch' },
      paraphrased: { subject: 'tone source', active: 'the tone source is on frequency', inactive: 'the tone source sounds noticeably flat' },
    },
    mirror_seated: {
      canonical: { subject: 'mirror shard', active: 'the mirror shard sits flush in its socket', inactive: 'the mirror socket is empty' },
      paraphrased: { subject: 'reflective insert', active: 'the reflective insert occupies the fitting', inactive: 'the reflective insert has not been installed' },
    },
    beacon_calibrated: {
      canonical: { subject: 'beacon', active: 'the beacon calibration cycle is complete', inactive: 'the beacon remains outside calibration' },
      paraphrased: { subject: 'receiver', active: 'the receiver is aligned to its reference', inactive: 'the receiver has not reached alignment' },
    },
    pressure_threshold_met: {
      canonical: { subject: 'pressure latch', active: 'pressure is above the latch mark', inactive: 'pressure remains below the latch mark' },
      paraphrased: { subject: 'relock trigger', active: 'the relock trigger has enough pressure to arm', inactive: 'the relock trigger lacks the pressure to arm' },
    },
  };
  const value = variants[id]?.[surface];
  if (!value) throw new Error(`V9 lacks ${surface} phrases for ${id} (${modelId}).`);
  return value;
}

function irrelevantPhrases(index: number, surface: V8SurfaceVariant, role: string): PhraseSet {
  const subject = surface === 'entity_renamed' ? role : [
    'wall marker',
    'floor indicator',
    'ceiling lamp',
    'archive dial',
  ][index % 4];
  return {
    subject,
    active: `${subject} shows its raised symbol`,
    inactive: `${subject} shows its lowered symbol`,
  };
}

function requiredOperator(mechanic: string): V9OperatorFamily {
  const value = OPERATOR_BY_MECHANIC[mechanic];
  if (!value) throw new Error(`V9 lacks an operator family for ${mechanic}.`);
  return value;
}

export const v9TemplateFamilies = TEMPLATE_BY_REPLICA;
export const v9OperatorByMechanic = OPERATOR_BY_MECHANIC;
