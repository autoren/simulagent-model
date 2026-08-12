import type { V8StructuredRecord, V8SurfaceVariant } from './contracts';
import type { V9BooleanValue, V9OperatorFamily, V9TemporalStatus } from './v9-contracts';
import type { V10GroundingRecord, V10Relation, V10Split, V10TemplateFamily } from './v10-contracts';
import { canonicalJson, shortHash } from './serialization';
import { evaluateAllowedTransitions } from './v9-symbolic';

interface PhraseSet {
  active: string;
  inactive: string;
}

type UncertaintyMode = 'unknown' | 'stale' | 'conflicting';

export const v10TemplateFamilies: V10TemplateFamily[] = [
  'direct_assertion',
  'explicit_negation',
  'denied_claim',
  'rejected_claim',
  'contrastive_correction',
  'scoped_rejection',
];

export const v10OperatorByMechanic: Record<string, V9OperatorFamily> = {
  hatch_traversal: 'binary_partition',
  beacon_calibration: 'binary_partition',
  pressure_hatch_relock: 'binary_partition',
  generator_tuning: 'multiway_partition',
  mirror_power_trip: 'multiway_partition',
  mirror_rejection: 'multiway_partition',
};

export function deriveV10AllowedValues(
  temporal: V9TemporalStatus,
  relations: [V10Relation, V10Relation],
): V9BooleanValue[] {
  if (temporal !== 'CURRENT') return ['inactive', 'active'];
  if (relations[0] === 'ENTAILED' && relations[1] === 'CONTRADICTED') return ['active'];
  if (relations[0] === 'CONTRADICTED' && relations[1] === 'ENTAILED') return ['inactive'];
  return ['inactive', 'active'];
}

export function buildV10GroundingRecords(
  sourceRecords: V8StructuredRecord[],
  sourceDatasetSha256: string,
  sourceReplica = 0,
): V10GroundingRecord[] {
  const base = sourceRecords.filter((record) => record.replica === sourceReplica);
  const complementSplits = complementSplitMap(base);
  const records = base.flatMap((source) => v10TemplateFamilies.map((template) =>
    createV10Record(source, sourceDatasetSha256, template, complementSplits.get(complementKey(source)) ?? 'train'),
  ));
  return records.sort((left, right) => canonicalJson([
    left.split,
    left.mechanic,
    left.context_group,
    left.template_family,
    left.intervention_member,
    left.state_lexicon_family,
  ]).localeCompare(canonicalJson([
    right.split,
    right.mechanic,
    right.context_group,
    right.template_family,
    right.intervention_member,
    right.state_lexicon_family,
  ])));
}

function complementSplitMap(records: V8StructuredRecord[]): Map<string, V10Split> {
  const byMechanic = new Map<string, Set<string>>();
  for (const record of records) {
    const values = byMechanic.get(record.mechanic) ?? new Set<string>();
    values.add(complementKey(record));
    byMechanic.set(record.mechanic, values);
  }
  const result = new Map<string, V10Split>();
  for (const values of byMechanic.values()) {
    const ordered = [...values].sort();
    ordered.forEach((value, index) => result.set(value, ordered.length > 1 && index === 0 ? 'evaluation' : 'train'));
  }
  return result;
}

function complementKey(record: V8StructuredRecord): string {
  const assignment = record.oracle.actual_assignment;
  const opposite = Object.fromEntries(Object.entries(assignment).map(([key, value]) => [key, !value]));
  const pair = [canonicalJson(assignment), canonicalJson(opposite)].sort();
  return `${record.mechanic}:${shortHash(pair, 24)}`;
}

function semanticContextKey(record: V8StructuredRecord): string {
  return canonicalJson([record.mechanic, record.primary_determinant_id, record.oracle.actual_assignment]);
}

function createV10Record(
  source: V8StructuredRecord,
  sourceDatasetSha256: string,
  template: V10TemplateFamily,
  split: V10Split,
): V10GroundingRecord {
  const schema = source.agent_input.action_dependency_schema;
  const relevantCount = schema.transition_determinants.length;
  const canonicalIds = source.target.determinant_ledger.slice(0, relevantCount).map((value) => value.id);
  const canonicalByModelId = new Map(schema.transition_determinants.map((value, index) => [value.id, canonicalIds[index]]));
  const phraseByModelId = new Map(schema.transition_determinants.map((value, index) => [
    value.id,
    determinantPhrases(canonicalIds[index], source.surface_variant, value.label),
  ]));
  const uncertaintyMode = uncertaintyModeFor(source);
  const rendered = source.agent_input.evidence_ledger.map((fact, index) => {
    const canonicalId = canonicalByModelId.get(fact.id);
    const phrases = canonicalId
      ? required(phraseByModelId, fact.id, 'determinant phrase')
      : irrelevantPhrases(index, source.surface_variant, fact.role);
    const temporal: V9TemporalStatus = fact.evidence_state === 'confirmed'
      ? 'CURRENT'
      : uncertaintyMode === 'stale'
        ? 'STALE_ONLY'
        : uncertaintyMode === 'conflicting'
          ? 'CONFLICTING_CURRENT'
          : 'UNKNOWN_CURRENT';
    const currentValue = fact.value === 'hidden' ? null : fact.value;
    const relations: [V10Relation, V10Relation] = currentValue === 'active'
      ? ['ENTAILED', 'CONTRADICTED']
      : currentValue === 'inactive'
        ? ['CONTRADICTED', 'ENTAILED']
        : ['UNKNOWN', 'UNKNOWN'];
    return {
      orderKey: shortHash([semanticContextKey(source), template, source.surface_variant, fact.id], 16),
      determinantId: canonicalId ? fact.id : null,
      phrases,
      temporal,
      currentValue,
      relations,
      allowedValues: deriveV10AllowedValues(temporal, relations),
      text: renderEvidence(phrases, currentValue, template, uncertaintyMode, index),
    };
  }).sort((left, right) => left.orderKey.localeCompare(right.orderKey));

  const evidenceUnits: V10GroundingRecord['evidence_units'] = [];
  let observation = '';
  for (const value of rendered) {
    if (observation) observation += '\n';
    const start = observation.length;
    observation += value.text;
    evidenceUnits.push({ start, end: observation.length, text: value.text });
  }
  const determinantGrounding = schema.transition_determinants.map((determinant) => {
    const index = rendered.findIndex((value) => value.determinantId === determinant.id);
    if (index < 0) throw new Error(`V10 rendering omitted determinant ${determinant.id}.`);
    const value = rendered[index];
    return {
      determinant_id: determinant.id,
      temporal_status: value.temporal,
      current_value: value.currentValue,
      hypothesis_relations: value.relations,
      allowed_values: value.allowedValues,
      evidence_span: evidenceUnits[index],
    };
  });
  const symbolic = evaluateAllowedTransitions({
    action_dependency_schema: schema,
    determinant_values: determinantGrounding,
  });
  const contextKey = semanticContextKey(source);
  return {
    id: `v10:${shortHash([source.id, template, split], 24)}`,
    schema_version: 10,
    split,
    context_group: `v10-context:${shortHash(contextKey, 24)}`,
    complement_group: `v10-complement:${shortHash(complementKey(source), 24)}`,
    intervention_group_id: `v10-intervention:${shortHash([contextKey, template], 24)}`,
    intervention_kind: source.intervention_kind,
    intervention_member: source.intervention_member,
    mechanic: source.mechanic,
    operator_family: required(v10OperatorByMechanic, source.mechanic, 'operator family'),
    template_family: template,
    state_lexicon_family: source.surface_variant,
    action_dependency_schema: schema,
    agent_input: {
      task: 'ground_current_state_polarity',
      candidate_action: schema.candidate_action,
      transition_determinants: schema.transition_determinants,
      state_hypotheses: schema.transition_determinants.map((determinant) => {
        const phrases = required(phraseByModelId, determinant.id, 'state hypotheses');
        return { determinant_id: determinant.id, statements: [phrases.active, phrases.inactive] };
      }),
      observation,
      output_instruction: 'Match each determinant to one evidence unit, classify its temporal status, and compare reliable current evidence with both supplied state hypotheses.',
    },
    evidence_units: evidenceUnits,
    target: {
      determinant_grounding: determinantGrounding,
      identifiable: symbolic.identifiable,
      possible_transition_codes: symbolic.possible_transition_codes,
    },
    source: { v8_record_id: source.id, v8_dataset_sha256: sourceDatasetSha256 },
  };
}

function uncertaintyModeFor(record: V8StructuredRecord): UncertaintyMode {
  const value = Number.parseInt(shortHash(semanticContextKey(record), 8), 16) % 3;
  return (['unknown', 'stale', 'conflicting'] as const)[value];
}

function renderEvidence(
  phrases: PhraseSet,
  value: V9BooleanValue | null,
  template: V10TemplateFamily,
  mode: UncertaintyMode,
  row: number,
): string {
  if (value === null) return renderUnresolved(phrases, mode, row);
  const actual = value === 'active' ? phrases.active : phrases.inactive;
  const opposite = value === 'active' ? phrases.inactive : phrases.active;
  switch (template) {
    case 'direct_assertion':
      return `A current verification establishes that ${actual}.`;
    case 'explicit_negation':
      return `A current verification establishes that it is not true that ${opposite}.`;
    case 'denied_claim':
      return `The current auditor denies the claim that ${opposite}.`;
    case 'rejected_claim':
      return `A signed present-time check rejects the report that ${opposite}.`;
    case 'contrastive_correction':
      return `The current correction says that ${opposite} is not the case; instead, ${actual}.`;
    case 'scoped_rejection':
      return `The report rejected by the current auditor is the one claiming that ${opposite}.`;
  }
}

function renderUnresolved(phrases: PhraseSet, mode: UncertaintyMode, row: number): string {
  if (mode === 'stale') {
    const earlier = row % 2 === 0 ? phrases.active : phrases.inactive;
    return `An archived reading said that ${earlier}, but no present reading is available.`;
  }
  if (mode === 'conflicting') {
    return `Two equally current readings conflict: one says that ${phrases.active}; the other says that ${phrases.inactive}.`;
  }
  return `No current evidence establishes either that ${phrases.active} or that ${phrases.inactive}.`;
}

function determinantPhrases(id: string, surface: V8SurfaceVariant, role: string): PhraseSet {
  if (surface === 'entity_renamed') {
    const subject = role.replace(/^the /, '');
    return { active: `${subject} is enabled`, inactive: `${subject} is disabled` };
  }
  const variants: Record<string, Record<'canonical' | 'paraphrased', PhraseSet>> = {
    hatch_unlocked: {
      canonical: { active: 'the observatory hatch stands unlatched', inactive: 'the observatory hatch remains latched' },
      paraphrased: { active: 'the upper passage can swing freely', inactive: 'the upper passage is held shut by its lock' },
    },
    generator_stable: {
      canonical: { active: 'the generator rhythm is even', inactive: 'the generator output surges unevenly' },
      paraphrased: { active: 'the coil bank carries steady power', inactive: 'the coil bank flickers between power levels' },
    },
    fork_calibrated: {
      canonical: { active: 'the fork tone matches the reference pitch', inactive: 'the fork tone falls away from the reference pitch' },
      paraphrased: { active: 'the tone source is on frequency', inactive: 'the tone source sounds noticeably flat' },
    },
    mirror_seated: {
      canonical: { active: 'the mirror shard sits flush in its socket', inactive: 'the mirror socket is empty' },
      paraphrased: { active: 'the reflective insert occupies the fitting', inactive: 'the reflective insert has not been installed' },
    },
    beacon_calibrated: {
      canonical: { active: 'the beacon calibration cycle is complete', inactive: 'the beacon remains outside calibration' },
      paraphrased: { active: 'the receiver is aligned to its reference', inactive: 'the receiver has not reached alignment' },
    },
    pressure_threshold_met: {
      canonical: { active: 'pressure is above the latch mark', inactive: 'pressure remains below the latch mark' },
      paraphrased: { active: 'the relock trigger has enough pressure to arm', inactive: 'the relock trigger lacks the pressure to arm' },
    },
  };
  const value = variants[id]?.[surface as 'canonical' | 'paraphrased'];
  if (!value) throw new Error(`V10 lacks ${surface} phrases for ${id}.`);
  return value;
}

function irrelevantPhrases(index: number, surface: V8SurfaceVariant, role: string): PhraseSet {
  const subject = surface === 'entity_renamed' ? role : [
    'wall marker',
    'floor indicator',
    'ceiling lamp',
    'archive dial',
  ][index % 4];
  return { active: `${subject} shows its raised symbol`, inactive: `${subject} shows its lowered symbol` };
}

function required<K, V>(values: Map<K, V>, key: K, label: string): V;
function required<V>(values: Record<string, V>, key: string, label: string): V;
function required<K, V>(values: Map<K, V> | Record<string, V>, key: K | string, label: string): V {
  const value = values instanceof Map ? values.get(key as K) : values[key as string];
  if (value === undefined) throw new Error(`Missing ${label}: ${String(key)}`);
  return value;
}
