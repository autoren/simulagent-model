import { describe, expect, it } from 'vitest';
import type { V7IdentifiabilityRecord } from '../src/contracts';
import { actionTemplate } from '../src/v7';
import { conditionalGapReport } from '../src/v7-validation';

function record(
  mechanic: 'relockshort' | 'powertrip',
  evidence: 'forced' | 'announced',
  action: string,
  surface: 'canonical' | 'entity_renamed',
  ambiguous: boolean,
): V7IdentifiabilityRecord {
  return {
    id: `${mechanic}:${evidence}:${action}:${surface}:${ambiguous}`,
    schema_version: 7,
    split: 'train',
    split_group: 'group',
    base_record_id: 'base',
    base_context_group: 'context',
    surface_pair_id: 'surface',
    surface_variant: surface,
    invariance_group_id: 'surface',
    evidence_intervention_id: 'evidence',
    evidence_intervention_kind: 'causal_rule_invariance',
    evidence_variant: evidence,
    mechanic,
    action_template: action,
    scenario_seeds: [9501],
    source_scenario_ids: [],
    agent_input: {} as V7IdentifiabilityRecord['agent_input'],
    target: { ambiguous, invariance: 'same_label_across_surfaces' },
  };
}

describe('V7 causal-evidence contract', () => {
  it('normalizes action keys into stable causal templates', () => {
    expect(actionTemplate('use:tuningFork')).toBe('use:tone');
    expect(actionTemplate('use:mirrorShard')).toBe('use:mirror');
    expect(actionTemplate('inspect:coilBank')).toBe('inspect:power');
    expect(actionTemplate('move:up')).toBe('move:up');
  });

  it('reports zero conditional gap only when every full cell is label-balanced', () => {
    const balanced = (['relockshort', 'powertrip'] as const).flatMap((mechanic) =>
      (['forced', 'announced'] as const).flatMap((evidence) =>
        (['use:tone', 'move:up'] as const).flatMap((action) =>
          (['canonical', 'entity_renamed'] as const).flatMap((surface) => [
            record(mechanic, evidence, action, surface, false),
            record(mechanic, evidence, action, surface, true),
          ]),
        ),
      ),
    );
    expect(conditionalGapReport(balanced).maximum).toBe(0);

    const unbalanced = balanced.filter((value, index) =>
      index !== 0 || value.target.ambiguous,
    );
    expect(conditionalGapReport(unbalanced).maximum).toBeGreaterThan(0);
  });
});
