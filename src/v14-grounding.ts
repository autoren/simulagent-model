import type { V8StructuredRecord, V8SurfaceVariant } from './contracts';
import type { V9BooleanValue, V9TemporalStatus } from './v9-contracts';
import type { V10Relation } from './v10-contracts';
import type { V14GroundingRecord, V14SemanticOperator, V14Split, V14SurfaceFamily } from './v14-contracts';
import { canonicalJson, shortHash } from './serialization';
import { evaluateAllowedTransitions } from './v9-symbolic';
import { deriveV10AllowedValues, v10OperatorByMechanic } from './v10-grounding';

interface PhraseSet { active: string; inactive: string }
type UncertaintyMode = 'unknown' | 'stale' | 'conflicting';

export const v14SurfacesByOperator: Record<V14SemanticOperator, V14SurfaceFamily[]> = {
  affirmative_gold: ['direct_assertion', 'present_confirmation', 'current_observation'],
  negated_opposite: ['explicit_negation', 'denied_claim', 'scoped_rejection'],
  contrastive_both: ['contrastive_correction', 'contrastive_verification', 'contrastive_resolution'],
};

export const v14SurfaceFamilies = Object.values(v14SurfacesByOperator).flat();

export const v14OperatorBySurface = Object.fromEntries(
  Object.entries(v14SurfacesByOperator).flatMap(([operator, surfaces]) =>
    surfaces.map((surface) => [surface, operator]),
  ),
) as Record<V14SurfaceFamily, V14SemanticOperator>;

export function buildV14GroundingRecords(
  sourceRecords: V8StructuredRecord[], sourceDatasetSha256: string, sourceReplica = 0,
): V14GroundingRecord[] {
  const base = sourceRecords.filter((record) => record.replica === sourceReplica);
  const splits = complementSplitMap(base);
  return base.flatMap((source) => v14SurfaceFamilies.map((surface) => createRecord(
    source, sourceDatasetSha256, surface, splits.get(complementKey(source)) ?? 'train',
  ))).sort((left, right) => canonicalJson([
    left.split, left.mechanic, left.context_group, left.semantic_operator_family,
    left.template_family, left.intervention_member, left.state_lexicon_family,
  ]).localeCompare(canonicalJson([
    right.split, right.mechanic, right.context_group, right.semantic_operator_family,
    right.template_family, right.intervention_member, right.state_lexicon_family,
  ])));
}

function complementSplitMap(records: V8StructuredRecord[]): Map<string, V14Split> {
  const byMechanic = new Map<string, Set<string>>();
  for (const record of records) {
    const values = byMechanic.get(record.mechanic) ?? new Set<string>();
    values.add(complementKey(record));
    byMechanic.set(record.mechanic, values);
  }
  const result = new Map<string, V14Split>();
  for (const values of byMechanic.values()) {
    [...values].sort().forEach((value, index, ordered) =>
      result.set(value, ordered.length > 1 && index === 0 ? 'evaluation' : 'train'),
    );
  }
  return result;
}

function complementKey(record: V8StructuredRecord): string {
  const assignment = record.oracle.actual_assignment;
  const opposite = Object.fromEntries(Object.entries(assignment).map(([key, value]) => [key, !value]));
  return `${record.mechanic}:${shortHash([canonicalJson(assignment), canonicalJson(opposite)].sort(), 24)}`;
}

function contextKey(record: V8StructuredRecord): string {
  return canonicalJson([record.mechanic, record.primary_determinant_id, record.oracle.actual_assignment]);
}

function createRecord(
  source: V8StructuredRecord, sourceDatasetSha256: string, surface: V14SurfaceFamily, split: V14Split,
): V14GroundingRecord {
  const schema = source.agent_input.action_dependency_schema;
  const relevantCount = schema.transition_determinants.length;
  const canonicalIds = source.target.determinant_ledger.slice(0, relevantCount).map((value) => value.id);
  const canonicalByModelId = new Map(schema.transition_determinants.map((value, index) => [value.id, canonicalIds[index]]));
  const phraseByModelId = new Map(schema.transition_determinants.map((value, index) => [
    value.id, determinantPhrases(canonicalIds[index], source.surface_variant, value.label),
  ]));
  const uncertaintyMode = uncertaintyModeFor(source);
  const rendered = source.agent_input.evidence_ledger.map((fact, index) => {
    const canonicalId = canonicalByModelId.get(fact.id);
    const phrases = canonicalId
      ? required(phraseByModelId, fact.id, 'determinant phrase')
      : irrelevantPhrases(index, source.surface_variant, fact.role);
    const temporal: V9TemporalStatus = fact.evidence_state === 'confirmed'
      ? 'CURRENT'
      : uncertaintyMode === 'stale' ? 'STALE_ONLY'
        : uncertaintyMode === 'conflicting' ? 'CONFLICTING_CURRENT' : 'UNKNOWN_CURRENT';
    const currentValue = fact.value === 'hidden' ? null : fact.value;
    const relations: [V10Relation, V10Relation] = currentValue === 'active'
      ? ['ENTAILED', 'CONTRADICTED']
      : currentValue === 'inactive' ? ['CONTRADICTED', 'ENTAILED'] : ['UNKNOWN', 'UNKNOWN'];
    return {
      orderKey: shortHash([source.id, contextKey(source), surface, source.surface_variant, fact.id], 16),
      determinantId: canonicalId ? fact.id : null,
      temporal, currentValue, relations,
      allowedValues: deriveV10AllowedValues(temporal, relations),
      text: currentValue === null
        ? renderUnresolved(phrases, uncertaintyMode, index)
        : renderCurrent(phrases, currentValue, surface),
    };
  }).sort((left, right) => left.orderKey.localeCompare(right.orderKey));

  const evidenceUnits: V14GroundingRecord['evidence_units'] = [];
  let observation = '';
  for (const value of rendered) {
    if (observation) observation += '\n';
    const start = observation.length;
    observation += value.text;
    evidenceUnits.push({ start, end: observation.length, text: value.text });
  }
  const grounding = schema.transition_determinants.map((determinant) => {
    const index = rendered.findIndex((value) => value.determinantId === determinant.id);
    if (index < 0) throw new Error(`V14 rendering omitted determinant ${determinant.id}.`);
    const value = rendered[index];
    return {
      determinant_id: determinant.id, temporal_status: value.temporal,
      current_value: value.currentValue, hypothesis_relations: value.relations,
      allowed_values: value.allowedValues, evidence_span: evidenceUnits[index],
    };
  });
  const symbolic = evaluateAllowedTransitions({ action_dependency_schema: schema, determinant_values: grounding });
  const semantic = v14OperatorBySurface[surface];
  const semanticKey = contextKey(source);
  return {
    id: `v14:${shortHash([source.id, surface, split], 24)}`,
    schema_version: 14,
    split,
    context_group: `v14-context:${shortHash(semanticKey, 24)}`,
    complement_group: `v14-complement:${shortHash(complementKey(source), 24)}`,
    intervention_group_id: `v14-intervention:${shortHash([semanticKey, surface], 24)}`,
    intervention_kind: source.intervention_kind,
    intervention_member: source.intervention_member,
    mechanic: source.mechanic,
    operator_family: required(v10OperatorByMechanic, source.mechanic, 'transition operator'),
    semantic_operator_family: semantic,
    template_family: surface,
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
      determinant_grounding: grounding,
      identifiable: symbolic.identifiable,
      possible_transition_codes: symbolic.possible_transition_codes,
    },
    source: { v8_record_id: source.id, v8_dataset_sha256: sourceDatasetSha256 },
  };
}

function renderCurrent(phrases: PhraseSet, value: V9BooleanValue, surface: V14SurfaceFamily): string {
  const actual = value === 'active' ? phrases.active : phrases.inactive;
  const opposite = value === 'active' ? phrases.inactive : phrases.active;
  switch (surface) {
    case 'direct_assertion': return `A current verification establishes that ${actual}.`;
    case 'present_confirmation': return `The latest inspection confirms that ${actual}.`;
    case 'current_observation': return `The present reading shows that ${actual}.`;
    case 'explicit_negation': return `A current verification establishes that it is not true that ${opposite}.`;
    case 'denied_claim': return `The current auditor denies the claim that ${opposite}.`;
    case 'scoped_rejection': return `The report rejected by the current auditor is the one claiming that ${opposite}.`;
    case 'contrastive_correction': return `The current correction says that ${opposite} is not the case; instead, ${actual}.`;
    case 'contrastive_verification': return `The present check rules out that ${opposite} and confirms that ${actual}.`;
    case 'contrastive_resolution': return `Of the two possibilities, current evidence excludes that ${opposite} and supports that ${actual}.`;
  }
}

function uncertaintyModeFor(record: V8StructuredRecord): UncertaintyMode {
  return (['unknown', 'stale', 'conflicting'] as const)[Number.parseInt(shortHash(contextKey(record), 8), 16) % 3];
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
  if (!value) throw new Error(`V14 lacks ${surface} phrases for ${id}.`);
  return value;
}

function irrelevantPhrases(index: number, surface: V8SurfaceVariant, role: string): PhraseSet {
  const subject = surface === 'entity_renamed' ? role : ['wall marker', 'floor indicator', 'ceiling lamp', 'archive dial'][index % 4];
  return { active: `${subject} shows its raised symbol`, inactive: `${subject} shows its lowered symbol` };
}

function required<K, V>(values: Map<K, V>, key: K, label: string): V;
function required<V>(values: Record<string, V>, key: string, label: string): V;
function required<K, V>(values: Map<K, V> | Record<string, V>, key: K | string, label: string): V {
  const value = values instanceof Map ? values.get(key as K) : values[key as string];
  if (value === undefined) throw new Error(`Missing ${label}: ${String(key)}`);
  return value;
}
