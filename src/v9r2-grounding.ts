import type { V9GroundingRecord } from './v9-contracts';
import { shortHash } from './serialization';
import { validateV9Grounding, type V9ValidationResult } from './v9-validation';

export interface V9r2ValidationResult extends V9ValidationResult {
  synthetic_context_identifiers: number;
}

export function removeSyntheticSceneIdentifiers(records: V9GroundingRecord[]): V9GroundingRecord[] {
  return records.map((record) => {
    const newline = record.agent_input.observation.indexOf('\n');
    if (newline < 0 || !record.agent_input.observation.startsWith('Audit scene ')) {
      throw new Error(`V9r2 source record lacks its locked synthetic scene line: ${record.id}`);
    }
    const offset = newline + 1;
    const shiftSpan = <T extends { start: number; end: number; text: string }>(span: T): T => ({
      ...span,
      start: span.start - offset,
      end: span.end - offset,
    });
    return {
      ...record,
      id: `v9r2:${shortHash(record.id, 24)}`,
      agent_input: {
        ...record.agent_input,
        observation: record.agent_input.observation.slice(offset),
      },
      evidence_units: record.evidence_units.map(shiftSpan),
      target: {
        ...record.target,
        determinant_grounding: record.target.determinant_grounding.map((target) => ({
          ...target,
          evidence_span: shiftSpan(target.evidence_span),
        })),
      },
    };
  });
}

export function validateV9r2(records: V9GroundingRecord[]): V9r2ValidationResult {
  const base = validateV9Grounding(records);
  const identifiers = records.filter((record) => /\bAudit scene [0-9a-f]+\b/i.test(record.agent_input.observation)).length;
  const errors = [...base.errors];
  if (identifiers > 0) errors.push(`V9r2 retains ${identifiers} synthetic context identifiers.`);
  return { ...base, errors, synthetic_context_identifiers: identifiers };
}
