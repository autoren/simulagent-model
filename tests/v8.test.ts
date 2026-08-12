import { describe, expect, it } from 'vitest';
import type { DatasetV8Config } from '../src/contracts';
import { buildV8Records, v8MechanicOrder } from '../src/v8';
import { validateV8 } from '../src/v8-validation';

const config: DatasetV8Config = {
  outputDir: 'data/v8-test',
  mechanics: [...v8MechanicOrder],
  surfaceVariants: ['canonical', 'entity_renamed', 'paraphrased'],
  replicasPerAssignment: 2,
  calibrationModulo: 2,
  simulatorSeeds: {
    hatch_traversal: 11101,
    generator_tuning: 11102,
    beacon_calibration: 11103,
    mirror_power_trip: 11104,
    mirror_rejection: 11105,
    pressure_hatch_relock: 11106,
  },
  shortcutGates: {
    maximumMetadataWorstFoldBalancedAccuracy: 0.55,
    maximumUnigramWorstFoldBalancedAccuracy: 0.55,
    maximumCharacterNgramWorstFoldBalancedAccuracy: 0.6,
    maximumLengthWorstFoldBalancedAccuracy: 0.55,
    maximumUnigramWorstFoldAuc: 0.65,
    maximumCharacterNgramWorstFoldAuc: 0.65,
    maximumLengthWorstFoldAuc: 0.55,
  },
  protocol: {
    model: 'mlx-community/Qwen3.5-0.8B-4bit',
    feature: 'layer_06_mean',
    cValue: 10,
    seed: 0,
    maxSeqLength: 1024,
    bootstrapSamples: 2000,
    bootstrapSeed: 8819,
    gates: {
      minimumEveryFoldPairDifferenceAccuracy: 0.6,
      minimumMeanPairDifferenceAccuracy: 0.7,
      minimumEveryFoldPointwiseScoreDirection: 0.6,
    },
  },
};

describe('V8 structured intervention curriculum', () => {
  it('derives dense flip pairs and causal controls from simulator transitions', () => {
    const built = buildV8Records(config);
    expect(built.label_flip_groups).toBeGreaterThan(0);
    expect(built.same_label_control_groups).toBeGreaterThan(0);
    for (const mechanic of v8MechanicOrder) {
      expect(built.groups_by_mechanic[mechanic].label_flip).toBeGreaterThanOrEqual(2);
    }
    expect(built.records.some((record) => record.source_scenario_id.includes('tonedrift'))).toBe(false);
  });

  it('keeps pair token bags matched and all development partitions isolated', () => {
    const built = buildV8Records(config);
    const validation = validateV8({
      records: built.records,
      mechanics: config.mechanics,
      surfaces: config.surfaceVariants,
      minimumLabelFlipGroupsPerMechanic: 2,
    });
    expect(validation.errors).toEqual([]);
    expect(validation.mismatched_pair_token_bags).toBe(0);
    expect(validation.mismatched_pair_prompt_lengths).toBe(0);
    expect(validation.context_cross_split_overlaps).toBe(0);
    expect(validation.tone_drift_records).toBe(0);
  });

  it('includes both known-true and known-false resolutions in every mechanic', () => {
    const built = buildV8Records(config);
    const validation = validateV8({
      records: built.records,
      mechanics: config.mechanics,
      surfaces: config.surfaceVariants,
      minimumLabelFlipGroupsPerMechanic: 2,
    });
    for (const mechanic of v8MechanicOrder) {
      expect(validation.mechanics[mechanic].unresolved_true_flip_groups).toBeGreaterThan(0);
      expect(validation.mechanics[mechanic].unresolved_false_flip_groups).toBeGreaterThan(0);
    }
  });
});
