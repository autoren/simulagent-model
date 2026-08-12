import { describe, expect, it } from 'vitest';
import type { AgentEpistemicInput } from '../src/contracts';
import { evidenceVariant, transformSurface } from '../src/v5-challenge';

const input: AgentEpistemicInput = {
  task: 'predict_possible_transitions',
  goal: 'restore the beacon before the hour repeats',
  observation: {
    turn: 1,
    location: 'atrium',
    locationName: 'Glass Atrium',
    description: 'The generator is stable beside the observatory hatch.',
    sensory: [],
    exits: [],
    visibleObjects: [{ id: 'brassKey', name: 'brass key', portable: true }],
    characters: [{ id: 'archivist', name: 'archivist' }],
    inventory: [],
    beliefs: ['The brass key should open the observatory hatch.'],
    memories: ['The hatch has relocked.'],
    pressure: 1,
    signal: 0,
  },
  recent_history: [{ action: 'use brass key', outcome: 'The hatch has relocked.' }],
  candidate_action: { key: 'use:brassKey', label: 'use brass key' },
  available_actions: [{ key: 'wait', label: 'wait' }],
};

describe('V5 challenge surfaces', () => {
  it('renames structured and narrative entities consistently without mutating input', () => {
    const renamed = transformSurface(input, 'entity_renamed');
    expect(renamed.observation.location).toBe('rotunda');
    expect(renamed.observation.visibleObjects[0].id).toBe('copperKey');
    expect(renamed.candidate_action.key).toBe('use:copperKey');
    expect(renamed.observation.description).toContain('power bay');
    expect(input.observation.location).toBe('atrium');
  });

  it('paraphrases natural text and labels while preserving action keys', () => {
    const paraphrased = transformSurface(input, 'paraphrased');
    expect(paraphrased.goal).toContain('repeating hour resets');
    expect(paraphrased.candidate_action.label).toBe('apply brass key');
    expect(paraphrased.candidate_action.key).toBe('use:brassKey');
    expect(paraphrased.observation.memories[0]).toContain('locked itself again');
  });

  it('classifies generated evidence scenario names from most specific to general', () => {
    expect(
      evidenceVariant(['gen-behavior-announced-consequence-powertrip-8101-trap']),
    ).toBe('announced-consequence');
    expect(
      evidenceVariant([
        'gen-behavior-forced-relockshort-8101-trap',
        'gen-behavior-unobservable-relockshort-8101-control',
      ]),
    ).toBe('mixed');
  });
});
