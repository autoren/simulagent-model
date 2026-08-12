import type { V8ActionDependencySchema, V8SurfaceVariant } from './contracts';
import type { V9BooleanValue, V9EvidenceSpan, V9EvidenceUnit, V9OperatorFamily, V9TemporalStatus } from './v9-contracts';
import type { V10Relation } from './v10-contracts';

export type V14Split = 'train' | 'evaluation';

export type V14SemanticOperator = 'affirmative_gold' | 'negated_opposite' | 'contrastive_both';

export type V14SurfaceFamily =
  | 'direct_assertion'
  | 'present_confirmation'
  | 'current_observation'
  | 'explicit_negation'
  | 'denied_claim'
  | 'scoped_rejection'
  | 'contrastive_correction'
  | 'contrastive_verification'
  | 'contrastive_resolution';

export interface V14GroundingTarget {
  determinant_id: string;
  temporal_status: V9TemporalStatus;
  current_value: V9BooleanValue | null;
  hypothesis_relations: [V10Relation, V10Relation];
  allowed_values: V9BooleanValue[];
  evidence_span: V9EvidenceSpan;
}

export interface V14GroundingRecord {
  id: string;
  schema_version: 14;
  split: V14Split;
  context_group: string;
  complement_group: string;
  intervention_group_id: string;
  intervention_kind: 'oracle_label_flip' | 'same_label_causal_control';
  intervention_member: 'relevant_unresolved' | 'relevant_resolved';
  mechanic: string;
  operator_family: V9OperatorFamily;
  semantic_operator_family: V14SemanticOperator;
  template_family: V14SurfaceFamily;
  state_lexicon_family: V8SurfaceVariant;
  action_dependency_schema: V8ActionDependencySchema;
  agent_input: {
    task: 'ground_current_state_polarity';
    candidate_action: string;
    transition_determinants: V8ActionDependencySchema['transition_determinants'];
    state_hypotheses: Array<{ determinant_id: string; statements: [string, string] }>;
    observation: string;
    output_instruction: string;
  };
  evidence_units: V9EvidenceUnit[];
  target: {
    determinant_grounding: V14GroundingTarget[];
    identifiable: boolean;
    possible_transition_codes: string[];
  };
  source: { v8_record_id: string; v8_dataset_sha256: string };
}
