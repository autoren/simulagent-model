import type {
  AgentTransitionInput,
  CounterfactualRecord,
  MlxExample,
  PrivilegedTransitionInput,
} from './contracts';
import { canonicalJson } from './serialization';

export const transitionSystemPrompt = [
  'Predict the exact next transition in a deterministic simulator.',
  'Use only the supplied state or observation history and candidate action.',
  'Do not choose a different action. Do not add explanation.',
  'Return one JSON object with exactly these fields: blocked_actions_added, blocked_actions_removed, environment_changed, flags_changed, hidden_actions_concealed, hidden_actions_revealed, inventory_added, inventory_removed, next_location, reachable_room_delta, success, visible_actions_added, visible_actions_removed.',
  'Use empty arrays and an empty flags_changed object when nothing changes.',
].join(' ');

export function toMlxExample(
  record: CounterfactualRecord,
  track: 'agent' | 'privileged',
): MlxExample {
  const input: AgentTransitionInput | PrivilegedTransitionInput =
    track === 'agent' ? record.agent_input : record.privileged_input;
  return {
    messages: [
      { role: 'system', content: transitionSystemPrompt },
      { role: 'user', content: canonicalJson(input) },
      { role: 'assistant', content: canonicalJson(record.target) },
    ],
  };
}
