import type {
  Action,
  ActualOutcomeSnapshot,
  Observation,
  ScenarioFamilyId,
  WorldFlags,
  WorldState,
} from '../../simulagent/src/simulation';

export type DatasetSplit = 'train' | 'valid' | 'test';
export type DatasetTrack = 'agent' | 'privileged';

export interface SplitRatios {
  train: number;
  valid: number;
  test: number;
}

export interface DatasetConfig {
  outputDir: string;
  scenarioIds: string[] | '*';
  maxStatesPerScenario: number;
  maxDepth: number;
  splitSeed: string;
  splitRatios: SplitRatios;
}

export interface DatasetV2Config {
  inputDir: string;
  outputDir: string;
  splitSeed: string;
  splitRatios: SplitRatios;
}

export interface DatasetV3Config extends DatasetV2Config {
  stratificationRestarts: number;
  minimumMechanicSupport: number;
  maximumAmbiguityRateGap: number;
  maximumMechanicShareGap: number;
}

export interface DatasetV4Config {
  inputDir: string;
  outputDir: string;
  splitSeed: string;
  calibrationRatio: number;
  stratificationRestarts: number;
  maximumAmbiguityRateGap: number;
  maximumMechanicShareGap: number;
}

export type V5ChallengeMechanic = 'relockshort' | 'powertrip';
export type V5EvidenceVariant =
  | 'forced'
  | 'announced'
  | 'announced-upstream'
  | 'announced-consequence'
  | 'announced-procedure'
  | 'unobservable'
  | 'mixed';
export type V5SurfaceVariant = 'canonical' | 'entity_renamed' | 'paraphrased';

export interface DatasetV5ChallengeConfig {
  outputDir: string;
  sourceDevelopmentDir: string;
  frozenProbeLock: string;
  scenarioSeeds: number[];
  mechanics: V5ChallengeMechanic[];
  evidenceVariants: Exclude<V5EvidenceVariant, 'mixed'>[];
  surfaceVariants: V5SurfaceVariant[];
  maxStatesPerScenario: number;
  maxDepth: number;
  evaluationGates: {
    minimumCanonicalBalancedAccuracy: number;
    minimumPerMechanicBalancedAccuracy: number;
    minimumSurfaceBalancedAccuracy: number;
    minimumSurfacePredictionAgreement: number;
    minimumEvidenceDirectionalAccuracy: number;
  };
}

export type V6Mechanic = 'relockshort' | 'powertrip' | 'mirrorreject';
export type V6DevelopmentMechanic = Exclude<V6Mechanic, 'mirrorreject'>;
export type V6Split = 'train' | 'calibration' | 'mechanic_holdout';

export interface DatasetV6Config {
  outputDir: string;
  priorDevelopmentDir: string;
  priorChallengeRecords: string;
  developmentSeeds: number[];
  holdoutSeeds: number[];
  developmentMechanics: V6DevelopmentMechanic[];
  holdoutMechanic: 'mirrorreject';
  evidenceVariants: Exclude<V5EvidenceVariant, 'mixed'>[];
  surfaceVariants: V5SurfaceVariant[];
  maxStatesPerScenario: number;
  maxDepth: number;
  calibrationRatio: number;
  stratificationRestarts: number;
  minimumEvidenceInterventionGroups: number;
  maximumAmbiguityRateGap: number;
  protocol: {
    model: string;
    feature: 'layer_06_mean';
    cValue: number;
    seed: number;
    maxSeqLength: number;
    bootstrapSamples: number;
    bootstrapSeed: number;
    referenceV5ChallengeBalancedAccuracy: number;
    gates: {
      minimumCalibrationCanonicalBalancedAccuracy: number;
      minimumHoldoutCanonicalBalancedAccuracy: number;
      minimumHoldoutBootstrapLowerBound: number;
      minimumSurfaceBalancedAccuracy: number;
      minimumSurfacePredictionAgreement: number;
      minimumCompleteTripletAccuracy: number;
      minimumAbsoluteImprovementOverV5: number;
      minimumEvidenceDirectionalAccuracy: number;
    };
  };
}

export interface ActionDescriptor {
  key: string;
  label: string;
}

export interface RecentTurn {
  action: string;
  outcome: string;
}

export interface AgentTransitionInput {
  task: 'predict_transition';
  goal: string;
  observation: Observation;
  recent_history: RecentTurn[];
  candidate_action: ActionDescriptor;
  available_actions: ActionDescriptor[];
}

export interface PrivilegedWorldSnapshot {
  turn: number;
  location: string;
  inventory: string[];
  flags: WorldFlags;
  rooms: WorldState['rooms'];
  pressure: number;
  signal: number;
  resolve: number;
}

export interface PrivilegedTransitionInput extends AgentTransitionInput {
  privileged_world_state: PrivilegedWorldSnapshot;
}

export interface ScenarioDynamicsSnapshot {
  storm_turn: number;
  generator_pressure_interval: number;
  hatch_relock_turn: number | null;
  hatch_relock_pressure: number;
  generator_trip_after_mirror_install: boolean;
  announce_environment_changes: boolean;
  announce_hatch_relock_upstream: boolean;
  announce_hatch_relock_consequence: boolean;
  announce_hatch_relock_procedure: boolean;
  announce_power_trip_consequence: boolean;
  announce_power_trip_procedure: boolean;
  hide_blocked_exit_signals: boolean;
  hide_power_state_signals: boolean;
}

export interface PrivilegedTransitionInputV2 extends PrivilegedTransitionInput {
  transition_rules: ScenarioDynamicsSnapshot;
}

export interface AgentEpistemicInput extends Omit<AgentTransitionInput, 'task'> {
  task: 'predict_possible_transitions';
}

export interface AgentOutcomeCountInput extends Omit<AgentTransitionInput, 'task'> {
  task: 'count_possible_transitions';
}

export interface AgentOutcomeCountTarget {
  outcome_count: number;
}

export interface TransitionTarget {
  success: boolean;
  next_location: string;
  inventory_added: string[];
  inventory_removed: string[];
  flags_changed: Partial<WorldFlags>;
  visible_actions_added: string[];
  visible_actions_removed: string[];
  blocked_actions_added: string[];
  blocked_actions_removed: string[];
  hidden_actions_revealed: string[];
  hidden_actions_concealed: string[];
  reachable_room_delta: number;
  environment_changed: boolean;
}

export interface CounterfactualRecord {
  id: string;
  schema_version: 1;
  split: DatasetSplit;
  split_group: string;
  scenario_id: string;
  scenario_family: ScenarioFamilyId;
  scenario_tags: string[];
  state_id: string;
  depth: number;
  action: Action;
  agent_input: AgentTransitionInput;
  privileged_input: PrivilegedTransitionInput;
  target: TransitionTarget;
  oracle_trace: {
    outcome_text: string;
    actual_outcome: ActualOutcomeSnapshot;
  };
}

export interface AgentEpistemicTarget {
  identifiable: boolean;
  possible_outcomes: TransitionTarget[];
}

export interface AgentEpistemicRecord {
  id: string;
  schema_version: 2;
  split: DatasetSplit;
  split_group: string;
  agent_input: AgentEpistemicInput;
  target: AgentEpistemicTarget;
  empirical_support: Array<{
    target_sha256: string;
    count: number;
  }>;
  source_record_count: number;
  source_scenario_ids: string[];
}

export interface AgentEpistemicRecordV3
  extends Omit<AgentEpistemicRecord, 'id' | 'schema_version'> {
  id: string;
  schema_version: 3;
  mechanic_labels: string[];
}

export type V4DevelopmentSplit = 'train' | 'calibration' | 'validation';

export interface AgentIdentifiabilityRecordV4
  extends Omit<AgentEpistemicRecordV3, 'id' | 'schema_version' | 'split'> {
  id: string;
  schema_version: 4;
  split: V4DevelopmentSplit;
  source_split: DatasetSplit;
}

export interface V5ChallengeRecord {
  id: string;
  schema_version: 5;
  split: 'challenge';
  split_group: string;
  base_record_id: string;
  base_context_group: string;
  surface_pair_id: string;
  surface_variant: V5SurfaceVariant;
  evidence_pair_id: string | null;
  evidence_variant: V5EvidenceVariant;
  mechanic: V5ChallengeMechanic;
  scenario_seeds: number[];
  source_scenario_ids: string[];
  agent_input: AgentEpistemicInput;
  target: AgentEpistemicTarget;
}

export interface V6IdentifiabilityTarget {
  ambiguous: boolean;
  invariance: 'same_label_across_surfaces';
}

export interface V6IdentifiabilityRecord {
  id: string;
  schema_version: 6;
  split: V6Split;
  split_group: string;
  base_record_id: string;
  base_context_group: string;
  surface_pair_id: string;
  surface_variant: V5SurfaceVariant;
  invariance_group_id: string;
  evidence_intervention_id: string | null;
  evidence_variant: V5EvidenceVariant;
  mechanic: V6Mechanic;
  scenario_seeds: number[];
  source_scenario_ids: string[];
  agent_input: AgentEpistemicInput;
  target: V6IdentifiabilityTarget;
}

export type V7DevelopmentMechanic = 'relockshort' | 'powertrip';
export type V7Mechanic = V7DevelopmentMechanic | 'tonedrift';
export type V7Split = 'train' | 'calibration' | 'untouched_mechanic';

export interface DatasetV7Config {
  outputDir: string;
  developmentSeeds: number[];
  holdoutSeeds: number[];
  developmentMechanics: V7DevelopmentMechanic[];
  holdoutMechanic: 'tonedrift';
  evidenceVariants: Exclude<V5EvidenceVariant, 'mixed'>[];
  surfaceVariants: V5SurfaceVariant[];
  maxStatesPerScenario: number;
  maxDepth: number;
  calibrationRatio: number;
  stratificationRestarts: number;
  maximumPairsPerConditionalStratum: number;
  minimumLabelChangingDevelopmentGroups: number;
  maximumConditionalLabelGap: number;
  shortcutGates: {
    maximumMetadataBalancedAccuracy: number;
    maximumEvidenceTextBalancedAccuracy: number;
    maximumEvidenceTextAuc: number;
  };
  protocol: {
    model: string;
    feature: 'layer_06_mean';
    cValue: number;
    seed: number;
    maxSeqLength: number;
    bootstrapSamples: number;
    bootstrapSeed: number;
    gates: {
      minimumCalibrationCanonicalBalancedAccuracy: number;
      minimumHoldoutCanonicalBalancedAccuracy: number;
      minimumHoldoutBootstrapLowerBound: number;
      minimumSurfaceBalancedAccuracy: number;
      minimumSurfacePredictionAgreement: number;
      minimumCompleteTripletAccuracy: number;
      minimumEvidenceDirectionalAccuracy: number;
      minimumPairedScoreDirectionalAccuracy: number;
      minimumWorstStratumBalancedAccuracy: number;
    };
  };
}

export interface V7IdentifiabilityTarget {
  ambiguous: boolean;
  invariance: 'same_label_across_surfaces';
}

export interface V7IdentifiabilityRecord {
  id: string;
  schema_version: 7;
  split: V7Split;
  split_group: string;
  base_record_id: string;
  base_context_group: string;
  surface_pair_id: string;
  surface_variant: V5SurfaceVariant;
  invariance_group_id: string;
  evidence_intervention_id: string;
  evidence_intervention_kind: 'causal_rule_invariance' | 'oracle_label_change';
  evidence_variant: Exclude<V5EvidenceVariant, 'mixed'> | 'mixed';
  mechanic: V7Mechanic;
  action_template: string;
  scenario_seeds: number[];
  source_scenario_ids: string[];
  agent_input: AgentEpistemicInput;
  target: V7IdentifiabilityTarget;
}

export type V8Mechanic =
  | 'hatch_traversal'
  | 'generator_tuning'
  | 'beacon_calibration'
  | 'mirror_power_trip'
  | 'mirror_rejection'
  | 'pressure_hatch_relock';
export type V8Split = 'train' | 'calibration';
export type V8SurfaceVariant = 'canonical' | 'entity_renamed' | 'paraphrased';
export type V8EvidenceState = 'confirmed' | 'unresolved';
export type V8DeterminantStatus =
  | 'RESOLVED_TRUE'
  | 'RESOLVED_FALSE'
  | 'UNRESOLVED_OUTCOME_SENSITIVE'
  | 'UNRESOLVED_OUTCOME_INVARIANT'
  | 'IRRELEVANT';

export interface V8ActionDependencySchema {
  candidate_action: string;
  transition_determinants: Array<{
    id: string;
    label: string;
  }>;
  transition_cases: Array<{
    values: Array<'active' | 'inactive'>;
    transition_code: string;
  }>;
  rule: 'Only the listed determinant roles may change the transition for this action.';
}

export interface V8EvidenceFact {
  id: string;
  role: string;
  evidence_state: V8EvidenceState;
  value: 'active' | 'inactive' | 'hidden';
}

export interface V8StructuredInput extends Omit<AgentEpistemicInput, 'task'> {
  task: 'classify_transition_determinants';
  action_dependency_schema: V8ActionDependencySchema;
  evidence_ledger: V8EvidenceFact[];
  output_instruction: string;
  format_padding: string;
}

export interface V8DeterminantTarget {
  id: string;
  status: V8DeterminantStatus;
}

export interface V8StructuredRecord {
  id: string;
  schema_version: 8;
  split: V8Split;
  split_group: string;
  mechanic: V8Mechanic;
  action_template: string;
  intervention_group_id: string;
  intervention_kind: 'oracle_label_flip' | 'same_label_causal_control';
  intervention_member: 'relevant_unresolved' | 'relevant_resolved';
  primary_determinant_id: string;
  primary_resolved_value: boolean;
  surface_group_id: string;
  surface_variant: V8SurfaceVariant;
  replica: number;
  source_scenario_id: string;
  agent_input: V8StructuredInput;
  target: {
    ambiguous: boolean;
    possible_transition_count: number;
    determinant_ledger: V8DeterminantTarget[];
    decisive_unresolved_determinants: string[];
    invariance: 'same_target_across_surfaces';
  };
  oracle: {
    actual_assignment: Record<string, boolean>;
    compatible_assignments: number;
    possible_transition_sha256: string[];
  };
}

export interface DatasetV8Config {
  outputDir: string;
  mechanics: V8Mechanic[];
  surfaceVariants: V8SurfaceVariant[];
  replicasPerAssignment: number;
  calibrationModulo: number;
  simulatorSeeds: Record<V8Mechanic, number>;
  shortcutGates: {
    maximumMetadataWorstFoldBalancedAccuracy: number;
    maximumUnigramWorstFoldBalancedAccuracy: number;
    maximumCharacterNgramWorstFoldBalancedAccuracy: number;
    maximumLengthWorstFoldBalancedAccuracy: number;
    maximumUnigramWorstFoldAuc: number;
    maximumCharacterNgramWorstFoldAuc: number;
    maximumLengthWorstFoldAuc: number;
  };
  protocol: {
    model: string;
    feature: 'layer_06_mean';
    cValue: number;
    seed: number;
    maxSeqLength: number;
    bootstrapSamples: number;
    bootstrapSeed: number;
    gates: {
      minimumEveryFoldPairDifferenceAccuracy: number;
      minimumMeanPairDifferenceAccuracy: number;
      minimumEveryFoldPointwiseScoreDirection: number;
    };
  };
}

export interface PrivilegedTransitionRecordV2 {
  id: string;
  schema_version: 2;
  split: DatasetSplit;
  split_group: string;
  source_split: DatasetSplit;
  source_split_group: string;
  scenario_id: string;
  scenario_family: ScenarioFamilyId;
  state_id: string;
  action: Action;
  agent_input: AgentTransitionInput;
  privileged_input: PrivilegedTransitionInputV2;
  target: TransitionTarget;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface MlxExample {
  messages: ChatMessage[];
}

export interface DatasetManifest {
  schema_version: 1;
  created_at: string;
  source_project: string;
  source_commit: string | null;
  config: DatasetConfig;
  counts: Record<DatasetSplit, number>;
  group_counts: Record<DatasetSplit, number>;
  scenario_counts: Record<string, number>;
  dataset_sha256: string;
}
