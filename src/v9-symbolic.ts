import type { V8StructuredInput } from './contracts';
import type {
  V9AllowedValues,
  V9BooleanValue,
  V9SymbolicInput,
  V9SymbolicResult,
} from './v9-contracts';

const VALUE_ORDER: V9BooleanValue[] = ['inactive', 'active'];

export function evaluateAllowedTransitions(input: V9SymbolicInput): V9SymbolicResult {
  const schema = input.action_dependency_schema;
  const determinantIds = schema.transition_determinants.map((value) => value.id);
  if (new Set(determinantIds).size !== determinantIds.length) {
    throw new Error('Transition schema contains duplicate determinant ids.');
  }
  const valuesById = new Map<string, V9BooleanValue[]>();
  for (const grounding of input.determinant_values) {
    if (!determinantIds.includes(grounding.determinant_id)) {
      throw new Error(`Grounding contains unknown determinant ${grounding.determinant_id}.`);
    }
    if (valuesById.has(grounding.determinant_id)) {
      throw new Error(`Grounding repeats determinant ${grounding.determinant_id}.`);
    }
    const values = normalizedValues(grounding.allowed_values);
    if (values.length === 0) {
      throw new Error(`Grounding has no allowed values for ${grounding.determinant_id}.`);
    }
    valuesById.set(grounding.determinant_id, values);
  }
  for (const id of determinantIds) {
    if (!valuesById.has(id)) throw new Error(`Grounding omits determinant ${id}.`);
  }

  const caseByAssignment = new Map<string, string>();
  for (const transitionCase of schema.transition_cases) {
    if (transitionCase.values.length !== determinantIds.length) {
      throw new Error('Transition case arity differs from its determinant schema.');
    }
    const key = assignmentKey(transitionCase.values);
    if (caseByAssignment.has(key)) throw new Error(`Transition schema repeats assignment ${key}.`);
    caseByAssignment.set(key, transitionCase.transition_code);
  }

  const compatible = cartesianProduct(
    determinantIds.map((id) => required(valuesById, id)),
  );
  const codes = compatible.map((assignment) => {
    const key = assignmentKey(assignment);
    const code = caseByAssignment.get(key);
    if (code === undefined) throw new Error(`Transition schema omits compatible assignment ${key}.`);
    return code;
  });
  const possible = [...new Set(codes)].sort();
  return {
    compatible_assignments: compatible.length,
    possible_transition_codes: possible,
    identifiable: possible.length === 1,
  };
}

export function allowedValuesFromV8Input(input: V8StructuredInput): V9AllowedValues[] {
  const evidenceById = new Map(input.evidence_ledger.map((fact) => [fact.id, fact]));
  return input.action_dependency_schema.transition_determinants.map((determinant) => {
    const evidence = evidenceById.get(determinant.id);
    if (!evidence) throw new Error(`V8 evidence omits determinant ${determinant.id}.`);
    if (evidence.evidence_state === 'unresolved') {
      return { determinant_id: determinant.id, allowed_values: [...VALUE_ORDER] };
    }
    if (evidence.value !== 'active' && evidence.value !== 'inactive') {
      throw new Error(`Confirmed V8 determinant ${determinant.id} has no Boolean value.`);
    }
    return { determinant_id: determinant.id, allowed_values: [evidence.value] };
  });
}

function normalizedValues(values: V9BooleanValue[]): V9BooleanValue[] {
  if (values.some((value) => value !== 'active' && value !== 'inactive')) {
    throw new Error('Allowed values must be active or inactive.');
  }
  return VALUE_ORDER.filter((value) => values.includes(value));
}

function cartesianProduct(values: V9BooleanValue[][]): V9BooleanValue[][] {
  return values.reduce<V9BooleanValue[][]>(
    (prefixes, next) => prefixes.flatMap((prefix) => next.map((value) => [...prefix, value])),
    [[]],
  );
}

function assignmentKey(values: readonly V9BooleanValue[]): string {
  return values.join('|');
}

function required<K, V>(values: Map<K, V>, key: K): V {
  const value = values.get(key);
  if (value === undefined) throw new Error(`Missing required map value ${String(key)}.`);
  return value;
}
