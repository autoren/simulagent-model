import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, test } from 'vitest';
import type { V8StructuredRecord } from '../src/contracts';
import { buildV14GroundingRecords, v14SurfaceFamilies, v14SurfacesByOperator } from '../src/v14-grounding';
import { currentMentionSignature, validateV14Grounding } from '../src/v14-validation';

function sourceRecords(): V8StructuredRecord[] {
  return ['train', 'calibration'].flatMap((split) =>
    readFileSync(resolve(`data/v8/records/${split}.jsonl`), 'utf8').trim().split('\n')
      .map((line) => JSON.parse(line) as V8StructuredRecord),
  );
}

describe('V14 operator-supported polarity corpus', () => {
  const records = buildV14GroundingRecords(sourceRecords(), 'test-v8-sha');

  test('creates the balanced complement-isolated corpus', () => {
    const validation = validateV14Grounding(records);
    expect(records).toHaveLength(4860);
    expect(validation.errors).toEqual([]);
    expect(validation.contexts.train + validation.contexts.evaluation).toBe(90);
    expect(validation.intervention_groups).toBe(810);
    expect(validation.current_hypothesis_pairs).toBe(11070);
    expect(validation.unsupported_surface_holdouts).toEqual([]);
  });

  test('provides three surface families for every semantic operator', () => {
    expect(v14SurfaceFamilies).toHaveLength(9);
    for (const surfaces of Object.values(v14SurfacesByOperator)) expect(surfaces).toHaveLength(3);
    expect(new Set(records.map((record) => record.template_family))).toEqual(new Set(v14SurfaceFamilies));
  });

  test('renders the registered mention orientation for every current target', () => {
    const expected = {
      affirmative_gold: 'gold_only',
      negated_opposite: 'opposite_only',
      contrastive_both: 'both',
    } as const;
    for (const record of records) {
      for (const target of record.target.determinant_grounding.filter((value) => value.temporal_status === 'CURRENT')) {
        expect(currentMentionSignature(record, target)).toBe(expected[record.semantic_operator_family]);
      }
    }
  });

  test('keeps exact spans and hides target labels', () => {
    for (const record of records) {
      expect(record.agent_input.observation).not.toMatch(/\b(?:active|inactive|entailed|contradicted|unknown)\b/i);
      for (const target of record.target.determinant_grounding) {
        expect(record.agent_input.observation.slice(target.evidence_span.start, target.evidence_span.end))
          .toBe(target.evidence_span.text);
      }
    }
  });
});
