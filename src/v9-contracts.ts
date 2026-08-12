import type { V8ActionDependencySchema } from './contracts';

export type V9BooleanValue = 'active' | 'inactive';

export interface V9AllowedValues {
  determinant_id: string;
  allowed_values: V9BooleanValue[];
}

export interface V9SymbolicResult {
  compatible_assignments: number;
  possible_transition_codes: string[];
  identifiable: boolean;
}

export interface V9SymbolicInput {
  action_dependency_schema: V8ActionDependencySchema;
  determinant_values: V9AllowedValues[];
}

export type V9TemporalStatus =
  | 'CURRENT'
  | 'UNKNOWN_CURRENT'
  | 'STALE_ONLY'
  | 'CONFLICTING_CURRENT';

export type V9TemplateFamily =
  | 'inspection_report'
  | 'operator_log'
  | 'questioned_claim'
  | 'technical_summary';

export type V9OperatorFamily = 'binary_partition' | 'multiway_partition';

export interface V9EvidenceSpan {
  start: number;
  end: number;
  text: string;
}

export interface V9GroundingTarget extends V9AllowedValues {
  temporal_status: V9TemporalStatus;
  evidence_span: V9EvidenceSpan;
}

export interface V9EvidenceUnit {
  start: number;
  end: number;
  text: string;
}

export interface V9GroundingInput {
  task: 'ground_transition_evidence';
  candidate_action: string;
  transition_determinants: V8ActionDependencySchema['transition_determinants'];
  observation: string;
  output_instruction: string;
}

export interface V9GroundingRecord {
  id: string;
  schema_version: 9;
  split: 'train' | 'calibration';
  context_group: string;
  intervention_group_id: string;
  intervention_kind: 'oracle_label_flip' | 'same_label_causal_control';
  intervention_member: 'relevant_unresolved' | 'relevant_resolved';
  mechanic: string;
  operator_family: V9OperatorFamily;
  template_family: V9TemplateFamily;
  surface_variant: 'canonical' | 'entity_renamed' | 'paraphrased';
  action_dependency_schema: V8ActionDependencySchema;
  agent_input: V9GroundingInput;
  evidence_units: V9EvidenceUnit[];
  target: {
    determinant_grounding: V9GroundingTarget[];
    identifiable: boolean;
    possible_transition_codes: string[];
  };
  source: {
    v8_record_id: string;
    v8_dataset_sha256: string;
  };
}
