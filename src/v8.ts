import type {
  Action,
  ObjectId,
  RoomId,
  ScenarioConfig,
  ScenarioVariantId,
  WorldState,
} from '../../simulagent/src/simulation';
import {
  createInitialState,
  generateBehavioralTrapScenarioCatalog,
  registerScenarioVariants,
  resolveAction,
} from '../../simulagent/src/simulation';
import type {
  DatasetV8Config,
  V8DeterminantStatus,
  V8Mechanic,
  V8StructuredInput,
  V8StructuredRecord,
  V8SurfaceVariant,
} from './contracts';
import { canonicalJson, sha256, shortHash } from './serialization';

type Assignment = Record<string, boolean>;

interface DeterminantDefinition {
  id: string;
  canonical: string;
  paraphrased: string;
}

interface MechanicDefinition {
  mechanic: V8Mechanic;
  action: Action;
  actionTemplate: string;
  actionLabels: Record<V8SurfaceVariant, string>;
  location: RoomId;
  determinants: DeterminantDefinition[];
  scenarioId: (scenarios: Map<V8Mechanic, ScenarioConfig>) => ScenarioVariantId;
  prepare: (state: WorldState, assignment: Assignment) => void;
}

export interface V8BuildResult {
  records: V8StructuredRecord[];
  intervention_groups: number;
  label_flip_groups: number;
  same_label_control_groups: number;
  groups_by_mechanic: Record<V8Mechanic, { total: number; label_flip: number; control: number }>;
  source_scenarios: Record<V8Mechanic, string>;
}

const MECHANIC_ORDER: V8Mechanic[] = [
  'hatch_traversal',
  'generator_tuning',
  'beacon_calibration',
  'mirror_power_trip',
  'mirror_rejection',
  'pressure_hatch_relock',
];

export function buildV8Records(config: DatasetV8Config): V8BuildResult {
  const scenarios = registerV8Scenarios(config);
  const definitions = mechanicDefinitions();
  const records: V8StructuredRecord[] = [];
  const groupKinds = new Map<string, V8StructuredRecord['intervention_kind']>();
  const sourceScenarios = {} as Record<V8Mechanic, string>;

  for (const mechanic of config.mechanics) {
    const definition = required(definitions, mechanic, 'mechanic definition');
    const sourceScenarioId = definition.scenarioId(scenarios);
    sourceScenarios[mechanic] = sourceScenarioId;
    const assignments = booleanAssignments(definition.determinants.map((value) => value.id));
    for (const assignment of assignments) {
      const assignmentId = shortHash(assignment, 16);
      for (const primary of definition.determinants) {
        const compatible = assignments.filter((candidate) =>
          definition.determinants.every((determinant) =>
            determinant.id === primary.id || candidate[determinant.id] === assignment[determinant.id],
          ),
        );
        const possibleHashes = uniqueSorted(
          compatible.map((candidate) => transitionHash(definition, sourceScenarioId, candidate)),
        );
        const unresolvedAmbiguous = possibleHashes.length > 1;
        for (let replica = 0; replica < config.replicasPerAssignment; replica += 1) {
          const groupId = `v8-intervention:${shortHash([
            mechanic,
            assignmentId,
            primary.id,
            replica,
          ], 24)}`;
          const kind: V8StructuredRecord['intervention_kind'] = unresolvedAmbiguous
            ? 'oracle_label_flip'
            : 'same_label_causal_control';
          groupKinds.set(groupId, kind);
          for (const member of ['relevant_unresolved', 'relevant_resolved'] as const) {
            const memberCompatible = member === 'relevant_unresolved' ? compatible : [assignment];
            const memberHashes = uniqueSorted(
              memberCompatible.map((candidate) => transitionHash(definition, sourceScenarioId, candidate)),
            );
            const baseId = `v8-base:${shortHash([groupId, member], 24)}`;
            for (const surface of config.surfaceVariants) {
              records.push(createRecord({
                definition,
                sourceScenarioId,
                assignment,
                primary,
                replica,
                groupId,
                baseId,
                member,
                memberHashes,
                compatibleAssignments: memberCompatible.length,
                surface,
                split: replica % config.calibrationModulo === 0 ? 'calibration' : 'train',
                kind,
              }));
            }
          }
        }
      }
    }
  }

  const groupsByMechanic = Object.fromEntries(MECHANIC_ORDER.map((mechanic) => [
    mechanic,
    { total: 0, label_flip: 0, control: 0 },
  ])) as V8BuildResult['groups_by_mechanic'];
  for (const [groupId, kind] of groupKinds) {
    const record = records.find((value) => value.intervention_group_id === groupId);
    if (!record) throw new Error(`Missing V8 intervention group ${groupId}.`);
    const counts = groupsByMechanic[record.mechanic];
    counts.total += 1;
    if (kind === 'oracle_label_flip') counts.label_flip += 1;
    else counts.control += 1;
  }
  return {
    records: records.sort(recordOrder),
    intervention_groups: groupKinds.size,
    label_flip_groups: [...groupKinds.values()].filter((value) => value === 'oracle_label_flip').length,
    same_label_control_groups: [...groupKinds.values()].filter((value) => value === 'same_label_causal_control').length,
    groups_by_mechanic: groupsByMechanic,
    source_scenarios: sourceScenarios,
  };
}

function createRecord(options: {
  definition: MechanicDefinition;
  sourceScenarioId: ScenarioVariantId;
  assignment: Assignment;
  primary: DeterminantDefinition;
  replica: number;
  groupId: string;
  baseId: string;
  member: 'relevant_unresolved' | 'relevant_resolved';
  memberHashes: string[];
  compatibleAssignments: number;
  surface: V8SurfaceVariant;
  split: 'train' | 'calibration';
  kind: V8StructuredRecord['intervention_kind'];
}): V8StructuredRecord {
  const {
    definition,
    sourceScenarioId,
    assignment,
    primary,
    replica,
    groupId,
    baseId,
    member,
    memberHashes,
    compatibleAssignments,
    surface,
    split,
    kind,
  } = options;
  const ambiguous = memberHashes.length > 1;
  const surfaceGroupId = `v8-surface:${shortHash([baseId, split], 24)}`;
  const contextId = `v8-context:${shortHash([
    definition.mechanic,
    shortHash(assignment, 16),
    primary.id,
    replica,
  ], 24)}`;
  return {
    id: `v8:${shortHash([surfaceGroupId, surface], 24)}`,
    schema_version: 8,
    split,
    split_group: contextId,
    mechanic: definition.mechanic,
    action_template: definition.actionTemplate,
    intervention_group_id: groupId,
    intervention_kind: kind,
    intervention_member: member,
    primary_determinant_id: primary.id,
    primary_resolved_value: assignment[primary.id],
    surface_group_id: surfaceGroupId,
    surface_variant: surface,
    replica,
    source_scenario_id: sourceScenarioId,
    agent_input: structuredInput(
      definition,
      sourceScenarioId,
      assignment,
      primary,
      replica,
      member,
      surface,
    ),
    target: {
      ambiguous,
      possible_transition_count: memberHashes.length,
      determinant_ledger: determinantTargets(definition, assignment, primary, member, ambiguous),
      decisive_unresolved_determinants: member === 'relevant_unresolved' && ambiguous
        ? [primary.id]
        : [],
      invariance: 'same_target_across_surfaces',
    },
    oracle: {
      actual_assignment: { ...assignment },
      compatible_assignments: compatibleAssignments,
      possible_transition_sha256: memberHashes,
    },
  };
}

function structuredInput(
  definition: MechanicDefinition,
  sourceScenarioId: ScenarioVariantId,
  assignment: Assignment,
  primary: DeterminantDefinition,
  replica: number,
  member: 'relevant_unresolved' | 'relevant_resolved',
  surface: V8SurfaceVariant,
): V8StructuredInput {
  const roleLabels = new Map(
    definition.determinants.map((determinant, index) => [
      determinant.id,
      determinantLabel(determinant, definition.mechanic, index, surface),
    ]),
  );
  const modelIds = new Map(
    definition.determinants.map((determinant, index) => [
      determinant.id,
      modelDeterminantId(determinant.id, index, surface),
    ]),
  );
  const evidence: V8StructuredInput['evidence_ledger'] = definition.determinants.map((determinant) => ({
    id: required(modelIds, determinant.id, 'surface determinant id'),
    role: required(roleLabels, determinant.id, 'surface role label'),
    evidence_state: determinant.id === primary.id && member === 'relevant_unresolved'
      ? 'unresolved' as const
      : 'confirmed' as const,
    value: determinant.id === primary.id && member === 'relevant_unresolved'
      ? 'hidden' as const
      : assignment[determinant.id] ? 'active' as const : 'inactive' as const,
  }));
  const confirmedDeterminants = evidence.filter((fact) => fact.evidence_state === 'confirmed');
  const activeNeeded = 3 - confirmedDeterminants.filter((fact) => fact.value === 'active').length;
  const inactiveNeeded = 3 - confirmedDeterminants.filter((fact) => fact.value === 'inactive').length;
  const fillerCount = 7 - definition.determinants.length;
  const fillerValues: Array<'active' | 'inactive'> = [
    ...Array.from({ length: activeNeeded }, () => 'active' as const),
    ...Array.from({ length: inactiveNeeded }, () => 'inactive' as const),
  ];
  const hiddenFiller = member === 'relevant_resolved';
  if (fillerValues.length + (hiddenFiller ? 1 : 0) !== fillerCount) {
    throw new Error(`V8 evidence balancing failed for ${definition.mechanic}.`);
  }
  for (let index = 0; index < fillerCount; index += 1) {
    const hidden = hiddenFiller && index === fillerCount - 1;
    evidence.push({
      id: irrelevantModelId(index, surface),
      role: irrelevantRole(index, surface),
      evidence_state: hidden ? 'unresolved' : 'confirmed',
      value: hidden ? 'hidden' : fillerValues[hiddenFiller && index >= fillerCount - 1 ? index - 1 : index],
    });
  }
  if (replica % 2 === 1) evidence.reverse();
  const actionLabel = definition.actionLabels[surface];
  const instruction = surface === 'paraphrased'
    ? 'For every evidence fact, report its determinant status and then decide whether all compatible worlds have one transition.'
    : 'Classify every evidence fact by determinant status, then report whether the candidate action has one transition across compatible worlds.';
  const result: V8StructuredInput = {
    task: 'classify_transition_determinants',
    goal: `Determine the epistemic transition status for audit context ${String(replica + 1).padStart(2, '0')}.`,
    observation: {
      turn: replica,
      location: definition.location,
      locationName: surface === 'entity_renamed' ? 'Test Chamber' : roomName(definition.location),
      description: 'The visible scene is held fixed while the evidence ledger records what is and is not established.',
      sensory: ['No unlisted sensory cue resolves a determinant.'],
      exits: [],
      visibleObjects: [],
      characters: [],
      inventory: [],
      beliefs: [],
      memories: ['Use the supplied dependency schema; do not infer an unstated transition rule.'],
      pressure: 0,
      signal: 0,
    },
    recent_history: [],
    candidate_action: { key: definition.actionTemplate, label: actionLabel },
    available_actions: [{ key: definition.actionTemplate, label: actionLabel }],
    action_dependency_schema: {
      candidate_action: actionLabel,
      transition_determinants: definition.determinants.map((determinant) => ({
        id: required(modelIds, determinant.id, 'dependency determinant id'),
        label: required(roleLabels, determinant.id, 'dependency role label'),
      })),
      transition_cases: transitionCases(definition, sourceScenarioId),
      rule: 'Only the listed determinant roles may change the transition for this action.',
    },
    evidence_ledger: evidence,
    output_instruction: instruction,
    format_padding: '',
  };
  const targetLength = 4096;
  const currentLength = canonicalJson(result).length;
  if (currentLength > targetLength) {
    throw new Error(`V8 input exceeds the fixed serialized length: ${currentLength}.`);
  }
  result.format_padding = 'x'.repeat(targetLength - currentLength);
  return result;
}

function modelDeterminantId(
  canonicalId: string,
  index: number,
  surface: V8SurfaceVariant,
): string {
  if (surface === 'canonical') return canonicalId;
  return surface === 'entity_renamed' ? `component_${aliasLetter(index)}` : `factor_${aliasLetter(index)}`;
}

function irrelevantModelId(index: number, surface: V8SurfaceVariant): string {
  if (surface === 'canonical') return `irrelevant_context_${index + 1}`;
  return surface === 'entity_renamed'
    ? `ambient_component_${aliasLetter(index)}`
    : `background_factor_${aliasLetter(index)}`;
}

function irrelevantRole(index: number, surface: V8SurfaceVariant): string {
  if (surface === 'canonical') return `ambient context marker ${index + 1}`;
  if (surface === 'entity_renamed') return `ambient marker ${aliasLetter(index)}`;
  return `unrelated background indicator ${index + 1}`;
}

function determinantTargets(
  definition: MechanicDefinition,
  assignment: Assignment,
  primary: DeterminantDefinition,
  member: 'relevant_unresolved' | 'relevant_resolved',
  ambiguous: boolean,
): V8StructuredRecord['target']['determinant_ledger'] {
  const result = definition.determinants.map((determinant) => ({
    id: determinant.id,
    status: determinant.id === primary.id && member === 'relevant_unresolved'
      ? (ambiguous
          ? 'UNRESOLVED_OUTCOME_SENSITIVE'
          : 'UNRESOLVED_OUTCOME_INVARIANT') as V8DeterminantStatus
      : (assignment[determinant.id] ? 'RESOLVED_TRUE' : 'RESOLVED_FALSE') as V8DeterminantStatus,
  }));
  for (let index = 0; index < 7 - definition.determinants.length; index += 1) {
    result.push({ id: `irrelevant_context_${index + 1}`, status: 'IRRELEVANT' });
  }
  return result;
}

function transitionHash(
  definition: MechanicDefinition,
  scenarioId: ScenarioVariantId,
  assignment: Assignment,
): string {
  const state = createInitialState(scenarioId);
  state.location = definition.location;
  state.rooms[definition.location].visited = true;
  state.agent.beliefs = [];
  state.agent.memories = [];
  state.stats.pressure = 0;
  state.stats.signal = 0;
  definition.prepare(state, assignment);
  const next = resolveAction(state, definition.action, { driver: 'manual' });
  const latest = next.log.at(-1);
  if (!latest?.actualOutcome || !latest.actionSurfaceDelta) {
    throw new Error(`V8 oracle transition missing for ${definition.mechanic}.`);
  }
  return sha256(canonicalJson({
    actual_outcome: latest.actualOutcome,
    action_surface_delta: latest.actionSurfaceDelta,
  }));
}

function transitionCases(
  definition: MechanicDefinition,
  scenarioId: ScenarioVariantId,
): V8StructuredInput['action_dependency_schema']['transition_cases'] {
  const assignments = booleanAssignments(definition.determinants.map((value) => value.id));
  const hashes = assignments.map((assignment) => transitionHash(definition, scenarioId, assignment));
  const codes = new Map(uniqueSorted(hashes).map((hash, index) => [
    hash,
    `transition_${String(index + 1).padStart(2, '0')}`,
  ]));
  return assignments.map((assignment, index) => ({
    values: definition.determinants.map((determinant) =>
      assignment[determinant.id] ? 'active' as const : 'inactive' as const,
    ),
    transition_code: required(codes, hashes[index], 'transition code'),
  }));
}

function mechanicDefinitions(): Map<V8Mechanic, MechanicDefinition> {
  const definitions: MechanicDefinition[] = [
    {
      mechanic: 'hatch_traversal',
      action: { type: 'move', direction: 'up' },
      actionTemplate: 'move:up',
      actionLabels: {
        canonical: 'move up through the observatory hatch',
        entity_renamed: 'cross portal alpha',
        paraphrased: 'attempt the upward passage',
      },
      location: 'atrium',
      determinants: [det('hatch_unlocked', 'the observatory hatch is unlocked', 'whether the upper passage is open')],
      scenarioId: () => 'baseline',
      prepare: (state, assignment) => {
        state.flags.doorUnlocked = assignment.hatch_unlocked;
      },
    },
    {
      mechanic: 'generator_tuning',
      action: { type: 'use', target: 'tuningFork' },
      actionTemplate: 'use:tone@generator',
      actionLabels: {
        canonical: 'use the tuning fork at the generator coils',
        entity_renamed: 'apply instrument alpha to station beta',
        paraphrased: 'sound the carried fork beside the coil bank',
      },
      location: 'generator',
      determinants: [
        det('generator_stable', 'the generator rhythm is stable', 'whether power is already steady'),
        det('fork_calibrated', 'the carried tuning fork is calibrated', 'whether the tone source is on frequency'),
      ],
      scenarioId: () => 'baseline',
      prepare: (state, assignment) => {
        give(state, 'tuningFork');
        state.flags.generatorStable = assignment.generator_stable;
        state.flags.tuningForkDetuned = !assignment.fork_calibrated;
      },
    },
    {
      mechanic: 'beacon_calibration',
      action: { type: 'use', target: 'tuningFork' },
      actionTemplate: 'use:tone@beacon',
      actionLabels: {
        canonical: 'use the tuning fork at the beacon console',
        entity_renamed: 'apply instrument alpha to station gamma',
        paraphrased: 'sound the carried fork at the observatory receiver',
      },
      location: 'observatory',
      determinants: [
        det('generator_stable', 'the generator rhythm is stable', 'whether steady power reaches the dome'),
        det('mirror_seated', 'the mirror shard is seated', 'whether the reflective insert occupies its socket'),
        det('fork_calibrated', 'the carried tuning fork is calibrated', 'whether the tone source is on frequency'),
      ],
      scenarioId: () => 'baseline',
      prepare: (state, assignment) => {
        give(state, 'tuningFork');
        state.flags.generatorStable = assignment.generator_stable;
        state.flags.mirrorShardInstalled = assignment.mirror_seated;
        state.flags.tuningForkDetuned = !assignment.fork_calibrated;
      },
    },
    {
      mechanic: 'mirror_power_trip',
      action: { type: 'use', target: 'mirrorShard' },
      actionTemplate: 'use:mirror@beacon',
      actionLabels: {
        canonical: 'install the mirror shard at the beacon console',
        entity_renamed: 'seat component alpha in station gamma',
        paraphrased: 'place the reflective shard into the observatory socket',
      },
      location: 'observatory',
      determinants: [
        det('generator_stable', 'the generator rhythm is stable', 'whether power is steady before installation'),
        det('beacon_calibrated', 'the beacon is already calibrated', 'whether calibration has already completed'),
        det('mirror_seated', 'the mirror shard is already seated', 'whether the reflective insert is already installed'),
      ],
      scenarioId: (scenarios) => required(scenarios, 'mirror_power_trip', 'power-trip scenario').id,
      prepare: (state, assignment) => {
        give(state, 'mirrorShard');
        state.flags.generatorStable = assignment.generator_stable;
        state.flags.beaconCalibrated = assignment.beacon_calibrated;
        state.flags.mirrorShardInstalled = assignment.mirror_seated;
        state.flags.generatorTripped = false;
      },
    },
    {
      mechanic: 'mirror_rejection',
      action: { type: 'use', target: 'mirrorShard' },
      actionTemplate: 'use:mirror@socket',
      actionLabels: {
        canonical: 'install the mirror shard in the rejecting socket',
        entity_renamed: 'seat component alpha in socket delta',
        paraphrased: 'place the reflective shard into the unstable fitting',
      },
      location: 'observatory',
      determinants: [
        det('beacon_calibrated', 'the beacon is already calibrated', 'whether calibration has already completed'),
        det('mirror_seated', 'the mirror shard is already seated', 'whether the reflective insert is already installed'),
      ],
      scenarioId: (scenarios) => required(scenarios, 'mirror_rejection', 'mirror-rejection scenario').id,
      prepare: (state, assignment) => {
        give(state, 'mirrorShard');
        state.flags.beaconCalibrated = assignment.beacon_calibrated;
        state.flags.mirrorShardInstalled = assignment.mirror_seated;
        state.flags.mirrorRejected = false;
      },
    },
    {
      mechanic: 'pressure_hatch_relock',
      action: { type: 'wait' },
      actionTemplate: 'wait:relock-window',
      actionLabels: {
        canonical: 'wait through the hatch relock window',
        entity_renamed: 'hold position through cycle epsilon',
        paraphrased: 'let one turn pass as the pressure latch updates',
      },
      location: 'atrium',
      determinants: [
        det('hatch_unlocked', 'the observatory hatch is unlocked', 'whether the upper passage begins open'),
        det('beacon_calibrated', 'the beacon is calibrated', 'whether the terminal objective is already complete'),
        det('pressure_threshold_met', 'pressure meets the relock threshold', 'whether the pressure latch is armed'),
      ],
      scenarioId: (scenarios) => required(scenarios, 'pressure_hatch_relock', 'relock scenario').id,
      prepare: (state, assignment) => {
        const threshold = state.scenario.hatchRelockPressure ?? 1;
        state.turn = Math.max(0, (state.scenario.hatchRelockTurn ?? 1) - 1);
        state.flags.doorUnlocked = assignment.hatch_unlocked;
        state.flags.beaconCalibrated = assignment.beacon_calibrated;
        state.flags.hatchRelocked = false;
        state.stats.pressure = assignment.pressure_threshold_met ? threshold + 1 : 0;
      },
    },
  ];
  return new Map(definitions.map((definition) => [definition.mechanic, definition]));
}

function registerV8Scenarios(config: DatasetV8Config): Map<V8Mechanic, ScenarioConfig> {
  const requests: Array<{ output: V8Mechanic; simulatorMechanic: 'powertrip' | 'mirrorreject' | 'relock' }> = [
    { output: 'mirror_power_trip', simulatorMechanic: 'powertrip' },
    { output: 'mirror_rejection', simulatorMechanic: 'mirrorreject' },
    { output: 'pressure_hatch_relock', simulatorMechanic: 'relock' },
  ];
  const result = new Map<V8Mechanic, ScenarioConfig>();
  for (const request of requests) {
    const generated = registerScenarioVariants(generateBehavioralTrapScenarioCatalog({
      seeds: [config.simulatorSeeds[request.output]],
      behavioralMechanics: [request.simulatorMechanic],
      behavioralVariants: ['forced'],
    }));
    const trap = generated.find((scenario) => scenario.id.endsWith('-trap'));
    if (!trap) throw new Error(`V8 failed to generate ${request.output} trap scenario.`);
    result.set(request.output, trap);
  }
  return result;
}

function booleanAssignments(ids: string[]): Assignment[] {
  return Array.from({ length: 2 ** ids.length }, (_, mask) => Object.fromEntries(
    ids.map((id, index) => [id, Boolean(mask & (1 << index))]),
  ));
}

function determinantLabel(
  determinant: DeterminantDefinition,
  mechanic: V8Mechanic,
  index: number,
  surface: V8SurfaceVariant,
): string {
  if (surface === 'canonical') return determinant.canonical;
  if (surface === 'paraphrased') return determinant.paraphrased;
  return `${mechanic.replaceAll('_', ' ')} component ${aliasLetter(index)}`;
}

function aliasLetter(index: number): string {
  return String.fromCharCode('alpha'.charCodeAt(0) + index);
}

function roomName(location: RoomId): string {
  const names: Record<RoomId, string> = {
    atrium: 'Resonance Atrium',
    archive: 'Index Archive',
    garden: 'Mist Garden',
    generator: 'Generator Annex',
    observatory: 'Beacon Observatory',
  };
  return names[location];
}

function give(state: WorldState, objectId: ObjectId): void {
  if (!state.inventory.includes(objectId)) state.inventory.push(objectId);
}

function det(id: string, canonical: string, paraphrased: string): DeterminantDefinition {
  return { id, canonical, paraphrased };
}

function uniqueSorted(values: string[]): string[] {
  return [...new Set(values)].sort();
}

function recordOrder(left: V8StructuredRecord, right: V8StructuredRecord): number {
  return canonicalJson([
    left.split,
    left.mechanic,
    left.intervention_group_id,
    left.intervention_member,
    left.surface_variant,
  ]).localeCompare(canonicalJson([
    right.split,
    right.mechanic,
    right.intervention_group_id,
    right.intervention_member,
    right.surface_variant,
  ]));
}

function required<K, V>(map: Map<K, V>, key: K, label: string): V {
  const value = map.get(key);
  if (value === undefined) throw new Error(`Missing V8 ${label}: ${String(key)}.`);
  return value;
}

export const v8MechanicOrder = MECHANIC_ORDER;
