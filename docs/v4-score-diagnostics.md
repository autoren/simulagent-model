# V4 binary score-resolution diagnostics

The selected A/B candidate logits were emitted at coarse numeric precision. Converting
them to Python floats after the forward pass does not recover ranking information that
was already quantized in the language-model output head.

| Seed | Step | Calibration unique margins | Validation unique margins | Validation range |
| ---: | ---: | ---: | ---: | --- |
| 0 | 200 | 2 | 2 | 0.2500 to 0.3750 |
| 1 | 300 | 3 | 3 | -0.1250 to 0.1250 |
| 2 | 200 | 2 | 2 | -0.5000 to -0.3750 |

Across selected seeds, each example was placed into only two or three margin bins.
This explains why fitted thresholds could change class proportions but could not
extract a fine-grained ordering. The next experiment should use a dedicated float32
classification head over hidden representations rather than subtracting two
low-precision vocabulary logits.

## Float32 direct-projection audit

The selected checkpoints were rescored without retraining by multiplying the final hidden state
by the A and B vocabulary rows in float32. This produced 153, 154, and 154 distinct validation
margins across seeds 0–2. Mean AUC improved from 0.506 to 0.540 and mean balanced accuracy from
50.14% to 51.16%, which is still near chance. The audit separates two findings: vocabulary-logit
precision did collapse the score bins, but the learned A/B projection itself still lacked a
stable ranking.
