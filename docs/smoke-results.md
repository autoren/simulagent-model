# Verified smoke result

Run on 2026-08-05 using an Apple M4 Pro with 64 GB unified memory.

## Stack

- Model: `mlx-community/Qwen3.5-4B-4bit`
- MLX-LM: 0.31.3
- MLX: 0.32.0
- Training track: pilot agent-view dataset
- Dataset SHA-256: `fd525cac0dc4e640f5f2c2c93c68ff522fa24f02e6b9b52a6338dd3e20044b4e`
- Trainable weights: 2.029M / 4,205.750M (0.048%)

## Five-step compatibility run

| Measurement | Result |
| --- | ---: |
| Final train loss | 0.088 |
| Final validation loss | 0.114 |
| Two-batch test loss | 0.524 |
| Two-batch test perplexity | 1.688 |
| Peak unified memory | 16.996 GB |

The full dataset was also token-length audited with the Qwen3.5 tokenizer. Agent-view examples
range from 492 to 794 tokens (95th percentile 740); privileged examples range from 815 to 1,118
tokens (95th percentile 1,064). The 1,536-token main-training cap preserves every example.

The saved adapter was loaded in a fresh inference process. With thinking mode disabled, it
generated valid JSON with the exact 13-field transition schema for both sampled held-out
records. The untuned model produced valid JSON but zero exact-schema outputs on the same two
records. Neither model achieved exact transition match in this tiny check.

These numbers establish pipeline compatibility only. Five optimizer steps and two generated
examples are not an accuracy experiment; use the 600-step configuration and complete held-out
generation for a substantive comparison.

## Qwen3.5-9B compatibility run

The matched four-layer, five-step configuration also completed successfully:

| Measurement | Result |
| --- | ---: |
| Trainable weights | 2.705M / 8,953.802M (0.030%) |
| Final train loss | 0.179 |
| Final validation loss | 0.198 |
| Two-batch test loss | 0.577 |
| Two-batch test perplexity | 1.780 |
| Peak unified memory | 19.690 GB |

The adapter reloaded in a fresh process and generated the exact output schema on both sampled
records, with zero exact transition matches. This validates 9B as a comfortable local training
tier; it does not resolve the ambiguity and input-overlap problems documented by the dataset
audit.
