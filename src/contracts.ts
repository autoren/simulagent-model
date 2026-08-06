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
