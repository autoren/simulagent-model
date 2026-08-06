import type {
  Action,
  ScenarioConfig,
  ScenarioVariantId,
  WorldState,
} from '../../simulagent/src/simulation';
import {
  actionKey,
  actionLabel,
  availableActions,
  createInitialState,
  makeObservation,
  resolveAction,
} from '../../simulagent/src/simulation';
import type {
  ActionDescriptor,
  AgentTransitionInput,
  CounterfactualRecord,
  DatasetSplit,
  PrivilegedTransitionInput,
  TransitionTarget,
} from './contracts';
import { canonicalJson, shortHash } from './serialization';
import { splitGroupForScenario } from './split';

interface CompileScenarioOptions {
  scenario: ScenarioConfig;
  split: DatasetSplit;
  maxStates: number;
  maxDepth: number;
}

interface QueueItem {
  state: WorldState;
  depth: number;
}

export function compileScenario(options: CompileScenarioOptions): CounterfactualRecord[] {
  const queue: QueueItem[] = [
    { state: createInitialState(options.scenario.id as ScenarioVariantId), depth: 0 },
  ];
  const visited = new Set<string>();
  const records: CounterfactualRecord[] = [];
  let stateCount = 0;

  while (queue.length > 0 && stateCount < options.maxStates) {
    const current = queue.shift();
    if (!current) {
      break;
    }
    const stateId = stateFingerprint(current.state);
    if (visited.has(stateId)) {
      continue;
    }
    visited.add(stateId);
    stateCount += 1;

    const actions = availableActions(current.state);
    for (const action of actions) {
      const next = resolveAction(current.state, action, { driver: 'manual' });
      records.push(
        createRecord({
          state: current.state,
          next,
          action,
          stateId,
          depth: current.depth,
          split: options.split,
        }),
      );
      if (current.depth < options.maxDepth) {
        queue.push({ state: next, depth: current.depth + 1 });
      }
    }
  }

  return records;
}

export function stateFingerprint(state: WorldState): string {
  return shortHash({
    scenario: state.scenario.id,
    turn: state.turn,
    location: state.location,
    inventory: [...state.inventory].sort(),
    flags: state.flags,
    rooms: state.rooms,
    stats: {
      pressure: state.stats.pressure,
      signal: state.stats.signal,
      resolve: state.stats.resolve,
    },
    beliefs: state.agent.beliefs.slice(-5),
    memories: state.agent.memories.slice(-5),
  });
}

function createRecord(options: {
  state: WorldState;
  next: WorldState;
  action: Action;
  stateId: string;
  depth: number;
  split: DatasetSplit;
}): CounterfactualRecord {
  const latest = options.next.log.at(-1);
  if (!latest?.actualOutcome || !latest.actionSurfaceDelta) {
    throw new Error(`Missing oracle transition for ${options.state.scenario.id}.`);
  }

  const descriptor = describeAction(options.action);
  const allActions = availableActions(options.state).map(describeAction);
  const agentInput: AgentTransitionInput = {
    task: 'predict_transition',
    goal: options.state.agent.goal,
    observation: makeObservation(options.state),
    recent_history: options.state.log.slice(-5).map((entry) => ({
      action: entry.action,
      outcome: entry.outcome,
    })),
    candidate_action: descriptor,
    available_actions: allActions,
  };
  const privilegedInput: PrivilegedTransitionInput = {
    ...agentInput,
    privileged_world_state: {
      turn: options.state.turn,
      location: options.state.location,
      inventory: [...options.state.inventory].sort(),
      flags: options.state.flags,
      rooms: options.state.rooms,
      pressure: options.state.stats.pressure,
      signal: options.state.stats.signal,
      resolve: options.state.stats.resolve,
    },
  };
  const target: TransitionTarget = {
    success: latest.actualOutcome.success,
    next_location: latest.actualOutcome.location,
    inventory_added: [...latest.actualOutcome.inventoryAdded].sort(),
    inventory_removed: [...latest.actualOutcome.inventoryRemoved].sort(),
    flags_changed: latest.actualOutcome.flagsChanged,
    visible_actions_added: [...latest.actionSurfaceDelta.addedVisibleActionKeys].sort(),
    visible_actions_removed: [...latest.actionSurfaceDelta.removedVisibleActionKeys].sort(),
    blocked_actions_added: [...latest.actionSurfaceDelta.addedBlockedActionKeys].sort(),
    blocked_actions_removed: [...latest.actionSurfaceDelta.removedBlockedActionKeys].sort(),
    hidden_actions_revealed: [...latest.actionSurfaceDelta.revealedHiddenActionKeys].sort(),
    hidden_actions_concealed: [...latest.actionSurfaceDelta.concealedHiddenActionKeys].sort(),
    reachable_room_delta: latest.actionSurfaceDelta.reachableRoomDelta,
    environment_changed: latest.actualOutcome.environmentChanged,
  };
  const actionId = actionKey(options.action);

  return {
    id: `${options.state.scenario.id}:${options.stateId}:${shortHash(actionId, 8)}`,
    schema_version: 1,
    split: options.split,
    split_group: splitGroupForScenario(options.state.scenario.id),
    scenario_id: options.state.scenario.id,
    scenario_family: options.state.scenario.family,
    scenario_tags: [...options.state.scenario.schemaTags].sort(),
    state_id: options.stateId,
    depth: options.depth,
    action: options.action,
    agent_input: agentInput,
    privileged_input: privilegedInput,
    target,
    oracle_trace: {
      outcome_text: latest.outcome,
      actual_outcome: latest.actualOutcome,
    },
  };
}

function describeAction(action: Action): ActionDescriptor {
  return {
    key: actionKey(action),
    label: actionLabel(action),
  };
}

export function recordSortKey(record: CounterfactualRecord): string {
  return canonicalJson([
    record.split,
    record.split_group,
    record.scenario_id,
    record.depth,
    record.state_id,
    record.agent_input.candidate_action.key,
  ]);
}

