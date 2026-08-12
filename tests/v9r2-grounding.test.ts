import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, test } from 'vitest';
import type { V9GroundingRecord } from '../src/v9-contracts';
import { removeSyntheticSceneIdentifiers, validateV9r2 } from '../src/v9r2-grounding';

function sourceRecords(): V9GroundingRecord[] {
  return ['train', 'calibration'].flatMap((split) =>
    readFileSync(resolve(`data/v9/records/${split}.jsonl`), 'utf8').trim().split('\n')
      .map((line) => JSON.parse(line) as V9GroundingRecord),
  );
}

describe('V9r2 context-identifier removal', () => {
  const source = sourceRecords();
  const records = removeSyntheticSceneIdentifiers(source);

  test('removes only the scene line and preserves all target semantics', () => {
    expect(records).toHaveLength(source.length);
    for (let index = 0; index < records.length; index += 1) {
      expect(records[index].agent_input.observation).not.toMatch(/^Audit scene/);
      expect(records[index].target.identifiable).toBe(source[index].target.identifiable);
      expect(records[index].target.possible_transition_codes).toEqual(source[index].target.possible_transition_codes);
      expect(records[index].target.determinant_grounding.map((value) => [
        value.determinant_id,
        value.allowed_values,
        value.temporal_status,
        value.evidence_span.text,
      ])).toEqual(source[index].target.determinant_grounding.map((value) => [
        value.determinant_id,
        value.allowed_values,
        value.temporal_status,
        value.evidence_span.text,
      ]));
    }
  });

  test('retains exact spans and split integrity without prompt duplicates', () => {
    const validation = validateV9r2(records);
    expect(validation.errors).toEqual([]);
    expect(validation.synthetic_context_identifiers).toBe(0);
    expect(validation.duplicate_prompts).toBe(0);
    expect(validation.cross_split_duplicate_prompts).toBe(0);
    expect(validation.malformed_spans).toBe(0);
  });
});
