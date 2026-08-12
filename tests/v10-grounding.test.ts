import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, test } from 'vitest';
import type { V8StructuredRecord } from '../src/contracts';
import { buildV10GroundingRecords, deriveV10AllowedValues, v10TemplateFamilies } from '../src/v10-grounding';
import { validateV10Grounding } from '../src/v10-validation';

function sourceRecords(): V8StructuredRecord[] {
  return ['train', 'calibration'].flatMap((split) =>
    readFileSync(resolve(`data/v8/records/${split}.jsonl`), 'utf8').trim().split('\n')
      .map((line) => JSON.parse(line) as V8StructuredRecord),
  );
}

describe('V10 polarity corpus', () => {
  const records = buildV10GroundingRecords(sourceRecords(), 'test-v8-sha');

  test('creates the locked balanced corpus and isolated complement split', () => {
    const validation = validateV10Grounding(records);
    expect(records).toHaveLength(3240);
    expect(validation.errors).toEqual([]);
    expect(validation.contexts.train + validation.contexts.evaluation).toBe(90);
    expect(validation.complement_cross_split_overlaps).toBe(0);
    expect(validation.imbalanced_current_cells).toBe(0);
    expect(validation.intervention_groups).toBe(540);
  });

  test('renders all six families with multiple negation constructions', () => {
    expect([...new Set(records.map((record) => record.template_family))].sort())
      .toEqual([...v10TemplateFamilies].sort());
    const currentText = Object.fromEntries(v10TemplateFamilies.map((template) => [
      template,
      records.find((record) => record.template_family === template &&
        record.target.determinant_grounding.some((target) => target.current_value !== null))!
        .agent_input.observation,
    ]));
    expect(currentText.direct_assertion).toContain('establishes');
    for (const template of v10TemplateFamilies.filter((value) => value !== 'direct_assertion')) {
      expect(currentText[template]).toMatch(/not|denies|reject/);
    }
  });

  test('derives allowed values from temporal status and complementary hypotheses', () => {
    expect(deriveV10AllowedValues('CURRENT', ['ENTAILED', 'CONTRADICTED'])).toEqual(['active']);
    expect(deriveV10AllowedValues('CURRENT', ['CONTRADICTED', 'ENTAILED'])).toEqual(['inactive']);
    expect(deriveV10AllowedValues('CURRENT', ['UNKNOWN', 'UNKNOWN'])).toEqual(['inactive', 'active']);
    expect(deriveV10AllowedValues('STALE_ONLY', ['ENTAILED', 'CONTRADICTED'])).toEqual(['inactive', 'active']);
    for (const record of records) {
      for (const target of record.target.determinant_grounding) {
        expect(target.allowed_values).toEqual(deriveV10AllowedValues(
          target.temporal_status,
          target.hypothesis_relations,
        ));
      }
    }
  });

  test('keeps exact spans and symbolic targets without literal observation labels', () => {
    for (const record of records) {
      expect(record.agent_input.observation).not.toMatch(/\b(?:active|inactive|entailed|contradicted|unknown)\b/i);
      for (const target of record.target.determinant_grounding) {
        expect(record.agent_input.observation.slice(target.evidence_span.start, target.evidence_span.end))
          .toBe(target.evidence_span.text);
      }
    }
  });
});
