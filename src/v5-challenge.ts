import type {
  AgentEpistemicInput,
  AgentEpistemicRecord,
  V5ChallengeMechanic,
  V5ChallengeRecord,
  V5EvidenceVariant,
  V5SurfaceVariant,
} from './contracts';
import { canonicalJson, shortHash } from './serialization';

const evidenceVariants: Exclude<V5EvidenceVariant, 'mixed'>[] = [
  'announced-consequence',
  'announced-procedure',
  'announced-upstream',
  'unobservable',
  'announced',
  'forced',
];

export interface V5ChallengeBuildResult {
  records: V5ChallengeRecord[];
  base_records: number;
  excluded_development_prompt_overlaps: number;
  evidence_pair_groups: number;
}

export function buildV5ChallengeRecords(options: {
  records: AgentEpistemicRecord[];
  developmentPromptKeys: Set<string>;
  surfaceVariants: V5SurfaceVariant[];
}): V5ChallengeBuildResult {
  const eligible = options.records.filter((record) => {
    const descriptor = sourceDescriptor(record.source_scenario_ids);
    return descriptor !== null && !options.developmentPromptKeys.has(canonicalJson(record.agent_input));
  });
  const excludedDevelopmentPromptOverlaps = options.records.filter(
    (record) => options.developmentPromptKeys.has(canonicalJson(record.agent_input)),
  ).length;
  const evidencePairs = evidencePairIds(eligible);
  const selected = new Map<string, AgentEpistemicRecord>();
  const byContext = groupBy(eligible, (record) => record.split_group);
  for (const contextRecords of byContext.values()) {
    const ambiguous = contextRecords
      .filter((record) => !record.target.identifiable)
      .sort(baseRecordOrder);
    const identifiable = contextRecords
      .filter((record) => record.target.identifiable)
      .sort(baseRecordOrder);
    if (ambiguous.length === 0 || identifiable.length === 0) continue;
    ambiguous.forEach((record, index) => {
      selected.set(record.id, record);
      selected.set(identifiable[index % identifiable.length].id, identifiable[index % identifiable.length]);
    });
  }
  for (const record of eligible) {
    if (evidencePairs.has(record.id)) selected.set(record.id, record);
  }
  const baseRecords = [...selected.values()].sort(baseRecordOrder);
  const records = baseRecords.flatMap((record) => {
    const descriptor = requiredSourceDescriptor(record.source_scenario_ids);
    const pairId = `v5-surface:${shortHash(record.agent_input, 24)}`;
    const evidencePairId = evidencePairs.get(record.id) ?? null;
    return options.surfaceVariants.map((surfaceVariant): V5ChallengeRecord => {
      const input = transformSurface(record.agent_input, surfaceVariant);
      return {
        id: `challenge-v5:${shortHash([pairId, surfaceVariant], 24)}`,
        schema_version: 5,
        split: 'challenge',
        split_group: `challenge-context:${shortHash(record.split_group, 24)}`,
        base_record_id: record.id,
        base_context_group: record.split_group,
        surface_pair_id: pairId,
        surface_variant: surfaceVariant,
        evidence_pair_id: evidencePairId,
        evidence_variant: evidenceVariant(record.source_scenario_ids),
        mechanic: descriptor.mechanic,
        scenario_seeds: descriptor.seeds,
        source_scenario_ids: [...record.source_scenario_ids],
        agent_input: input,
        target: record.target,
      };
    });
  }).sort(challengeRecordOrder);
  return {
    records,
    base_records: baseRecords.length,
    excluded_development_prompt_overlaps: excludedDevelopmentPromptOverlaps,
    evidence_pair_groups: new Set(records.flatMap((record) => record.evidence_pair_id ?? [])).size,
  };
}

export function evidenceVariant(sourceScenarioIds: string[]): V5EvidenceVariant {
  const values = new Set(
    sourceScenarioIds.map((id) => evidenceVariants.find((variant) => id.includes(`-${variant}-`))),
  );
  values.delete(undefined);
  return values.size === 1 ? ([...values][0] as V5EvidenceVariant) : 'mixed';
}

export function transformSurface(
  input: AgentEpistemicInput,
  variant: V5SurfaceVariant,
): AgentEpistemicInput {
  if (variant === 'canonical') return structuredClone(input);
  if (variant === 'entity_renamed') {
    return transformStrings(input, renameEntity) as AgentEpistemicInput;
  }
  const transformed = structuredClone(input);
  transformed.goal = paraphraseText(transformed.goal);
  transformed.observation.description = paraphraseText(transformed.observation.description);
  transformed.observation.sensory = transformed.observation.sensory.map(paraphraseText);
  transformed.observation.beliefs = transformed.observation.beliefs.map(paraphraseText);
  transformed.observation.memories = transformed.observation.memories.map(paraphraseText);
  transformed.recent_history = transformed.recent_history.map((turn) => ({
    action: paraphraseActionLabel(turn.action),
    outcome: paraphraseText(turn.outcome),
  }));
  transformed.candidate_action.label = paraphraseActionLabel(transformed.candidate_action.label);
  transformed.available_actions = transformed.available_actions.map((action) => ({
    ...action,
    label: paraphraseActionLabel(action.label),
  }));
  return transformed;
}

export function evidenceSignature(record: AgentEpistemicRecord): string | null {
  const descriptor = sourceDescriptor(record.source_scenario_ids);
  if (!descriptor) return null;
  const observation = record.agent_input.observation;
  return canonicalJson({
    mechanic: descriptor.mechanic,
    turn: observation.turn,
    location: observation.location,
    exits: observation.exits.map((exit) => [exit.direction, exit.blocked]),
    visible_objects: observation.visibleObjects.map((object) => object.id),
    characters: observation.characters.map((character) => character.id),
    inventory: observation.inventory.map((object) => object.id),
    pressure: observation.pressure,
    signal: observation.signal,
    candidate_action: record.agent_input.candidate_action.key,
    available_actions: record.agent_input.available_actions.map((action) => action.key),
  });
}

function evidencePairIds(records: AgentEpistemicRecord[]): Map<string, string> {
  const pairs = new Map<string, string>();
  const groups = groupBy(
    records.filter((record) => evidenceSignature(record) !== null),
    (record) => evidenceSignature(record) as string,
  );
  for (const [signature, values] of groups) {
    const labels = new Set(values.map((record) => record.target.identifiable));
    const variants = new Set(values.map((record) => evidenceVariant(record.source_scenario_ids)));
    if (labels.size < 2 || variants.size < 2) continue;
    const pairId = `v5-evidence:${shortHash(signature, 24)}`;
    for (const record of values) pairs.set(record.id, pairId);
  }
  return pairs;
}

function sourceDescriptor(
  sourceScenarioIds: string[],
): { mechanic: V5ChallengeMechanic; seeds: number[] } | null {
  const mechanics = new Set<V5ChallengeMechanic>();
  const seeds = new Set<number>();
  for (const id of sourceScenarioIds) {
    const match = id.match(/-((?:relockshort)|(?:powertrip))-(\d+)-(?:trap|control)$/);
    if (!match) return null;
    mechanics.add(match[1] as V5ChallengeMechanic);
    seeds.add(Number(match[2]));
  }
  if (mechanics.size !== 1 || seeds.size === 0) return null;
  return { mechanic: [...mechanics][0], seeds: [...seeds].sort((left, right) => left - right) };
}

function requiredSourceDescriptor(sourceScenarioIds: string[]) {
  const descriptor = sourceDescriptor(sourceScenarioIds);
  if (!descriptor) throw new Error(`Invalid V5 source scenarios: ${sourceScenarioIds.join(', ')}`);
  return descriptor;
}

function transformStrings(value: unknown, transform: (value: string) => string): unknown {
  if (typeof value === 'string') return transform(value);
  if (Array.isArray(value)) return value.map((entry) => transformStrings(entry, transform));
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, transformStrings(entry, transform)]),
    );
  }
  return value;
}

const exactEntityRenames: Record<string, string> = {
  Mira: 'Iona',
  atrium: 'rotunda',
  archive: 'recordsHall',
  garden: 'conservatory',
  generator: 'powerBay',
  observatory: 'skyChamber',
  brassKey: 'copperKey',
  tuningFork: 'resonanceRod',
  mirrorShard: 'prismFragment',
  mistFountain: 'fogBasin',
  coilBank: 'relayArray',
  beaconConsole: 'signalConsole',
  starLens: 'skyLens',
  plaque: 'inscription',
  archivist: 'curator',
  indexCabinet: 'catalogCase',
  looseAtlas: 'foldedChart',
};

const entityPhraseRenames: Array<[string, string]> = [
  ['Index Archive', 'Ledger Hall'],
  ['Glass Atrium', 'Glass Rotunda'],
  ['Mist Garden', 'Vapor Conservatory'],
  ['Generator Annex', 'Power Bay'],
  ['Observatory', 'Sky Chamber'],
  ['Mira', 'Iona'],
  ['brass key', 'copper key'],
  ['tuning fork', 'resonance rod'],
  ['mirror shard', 'prism fragment'],
  ['mist fountain', 'fog basin'],
  ['coil bank', 'relay array'],
  ['beacon console', 'signal console'],
  ['beacon', 'signal prism'],
  ['star lens', 'sky lens'],
  ['archivist', 'curator'],
  ['index cabinet', 'catalog case'],
  ['loose atlas', 'folded chart'],
  ['atrium', 'rotunda'],
  ['archive', 'records hall'],
  ['garden', 'conservatory'],
  ['generator', 'power bay'],
  ['observatory', 'sky chamber'],
];

function renameEntity(value: string): string {
  if (exactEntityRenames[value]) return exactEntityRenames[value];
  if (value.includes(':')) {
    return value
      .split(':')
      .map((part) => exactEntityRenames[part] ?? part)
      .join(':');
  }
  return entityPhraseRenames.reduce(
    (current, [source, replacement]) => current.split(source).join(replacement),
    value,
  );
}

const paraphrases: Array<[string, string]> = [
  ['restore the beacon before the hour repeats', 'repair the beacon before the repeating hour resets'],
  ['The world engine initializes the station.', 'The station simulation begins.'],
  ['Hidden state is held outside the protagonist view.', 'Latent conditions remain outside the agent view.'],
  ['has relocked', 'has locked itself again'],
  ['relocks', 'locks itself again'],
  ['is unlocked now', 'now stands open'],
  ['is stable', 'remains steady'],
  ['fell out of rhythm', 'lost its steady cadence'],
  ['must be re-stabilized', 'needs to be stabilized again'],
  ['will not hold', 'cannot remain stable'],
  ['The shard fits the mirrored socket.', 'The fragment seats inside the mirrored socket.'],
  ['The star lens swivels toward a hidden point.', 'The suspended lens turns toward an unseen coordinate.'],
  ['Power relay tripped', 'The power relay shut down'],
  ['Mira waits long enough to separate signal from noise.', 'Mira pauses until the signal can be distinguished from the noise.'],
  ['The fork rings once', 'The fork sounds a single note'],
  ['The hatch is unlocked', 'The hatch now stands open'],
  ['steady power', 'stable power'],
  ['move up', 'travel upward'],
  ['use the key', 'apply the key'],
  ['before climbing', 'prior to climbing'],
];

function paraphraseText(value: string): string {
  return paraphrases.reduce(
    (current, [source, replacement]) => current.split(source).join(replacement),
    value,
  );
}

function paraphraseActionLabel(value: string): string {
  return paraphraseText(value)
    .replace(/^inspect /, 'examine ')
    .replace(/^move /, 'travel ')
    .replace(/^talk to /, 'speak with ')
    .replace(/^take /, 'collect ')
    .replace(/^use /, 'apply ')
    .replace(/^wait$/, 'pause');
}

function groupBy<T>(values: T[], key: (value: T) => string): Map<string, T[]> {
  const groups = new Map<string, T[]>();
  for (const value of values) groups.set(key(value), [...(groups.get(key(value)) ?? []), value]);
  return groups;
}

function baseRecordOrder(left: AgentEpistemicRecord, right: AgentEpistemicRecord): number {
  return canonicalJson([left.split_group, left.id]).localeCompare(
    canonicalJson([right.split_group, right.id]),
  );
}

function challengeRecordOrder(left: V5ChallengeRecord, right: V5ChallengeRecord): number {
  return canonicalJson([left.split_group, left.base_record_id, left.surface_variant]).localeCompare(
    canonicalJson([right.split_group, right.base_record_id, right.surface_variant]),
  );
}
