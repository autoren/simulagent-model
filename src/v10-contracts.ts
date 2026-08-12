import type { V8ActionDependencySchema, V8SurfaceVariant } from './contracts';
import type { V9BooleanValue, V9EvidenceSpan, V9EvidenceUnit, V9OperatorFamily, V9TemporalStatus } from './v9-contracts';

export type V10Split = 'train' | 'evaluation';

export type V10TemplateFamily =
  | 'direct_assertion'
  | 'explicit_negation'
  | 'denied_claim'
  | 'rejected_claim'
  | 'contrastive_correction'
  | 'scoped_rejection';

export type V10Relation = 'ENTAILED' | 'CONTRADICTED' | 'UNKNOWN';

export interface V10StateHypotheses {
  determinant_id: string;
  statements: [string, string];
}

export interface V10GroundingTarget {
  determinant_id: string;
  temporal_status: V9TemporalStatus;
  current_value: V9BooleanValue | null;
  hypothesis_relations: [V10Relation, V10Relation];
  allowed_values: V9BooleanValue[];
  evidence_span: V9EvidenceSpan;
}

export interface V10GroundingRecord {
  id: string;
  schema_version: 10;
  split: V10Split;
  context_group: string;
  complement_group: string;
  intervention_group_id: string;
  intervention_kind: 'oracle_label_flip' | 'same_label_causal_control';
  intervention_member: 'relevant_unresolved' | 'relevant_resolved';
  mechanic: string;
  operator_family: V9OperatorFamily;
  template_family: V10TemplateFamily;
  state_lexicon_family: V8SurfaceVariant;
  action_dependency_schema: V8ActionDependencySchema;
  agent_input: {
    task: 'ground_current_state_polarity';
    candidate_action: string;
    transition_determinants: V8ActionDependencySchema['transition_determinants'];
    state_hypotheses: V10StateHypotheses[];
    observation: string;
    output_instruction: string;
  };
  evidence_units: V9EvidenceUnit[];
  target: {
    determinant_grounding: V10GroundingTarget[];
    identifiable: boolean;
    possible_transition_codes: string[];
  };
  source: {
    v8_record_id: string;
    v8_dataset_sha256: string;
  };
}
